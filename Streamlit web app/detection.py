# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import skimage
from PIL import Image, ImageColor
from ultralytics import YOLO
from sklearn.metrics import mean_squared_error
import os
import json
import yaml
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import random  # Added for commentary generation

def get_labels_dics():
    ## Get tactical map keypoints positions dictionary ##
    json_path = "../pitch map labels position.json"
    with open(json_path, 'r') as f:
        keypoints_map_pos = json.load(f)

    ## Get football field keypoints numerical to alphabetical mapping ##
    yaml_path = "../config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file)
    classes_names_dic = classes_names_dic['names']

    ## Get football field players numerical to alphabetical mapping ##
    yaml_path = "../config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB")
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")
    colors_dic = {
        team1_name: [team1_p_color_rgb, team1_gk_color_rgb],
        team2_name: [team2_p_color_rgb, team2_gk_color_rgb]
    }
    colors_list = colors_dic[team1_name] + colors_dic[team2_name]
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list]
    return colors_dic, color_list_lab

def generate_file_name():
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx += 1
        output_file_name = f'detect_{idx}'
        if output_file_name + '.mp4' not in list_video_files:
            break
    return output_file_name

# Add the generate_commentary function from the second detection.py
def generate_commentary(frame_nbr, possession_team, detected_ball_src_pos, team_names, team_a_positions, team_b_positions, tac_height, tac_width, ball_track_history, labels_p):
    commentary = []
    team1_name, team2_name = team_names

    # Define field zones and goal areas (adjusted for bottom-attacking orientation)
    defensive_zone = (0, tac_height // 3)  # Top third (defense)
    midfield_zone = (tac_height // 3, 2 * tac_height // 3)  # Middle third
    attacking_zone = (2 * tac_height // 3, tac_height)  # Bottom third (attack)
    goal_area = (tac_height - tac_height // 10, tac_height)  # Bottom 10% (attacking third)

    # Count players in each zone for the current frame
    curr_team_a_positions = team_a_positions[-len(pred_dst_pts):] if 'pred_dst_pts' in locals() and len(pred_dst_pts) > 0 else []
    curr_team_b_positions = team_b_positions[-len(pred_dst_pts):] if 'pred_dst_pts' in locals() and len(pred_dst_pts) > 0 else []
    
    team_a_defense = sum(1 for x, y in curr_team_a_positions if defensive_zone[0] <= y < defensive_zone[1])
    team_a_midfield = sum(1 for x, y in curr_team_a_positions if midfield_zone[0] <= y < midfield_zone[1])
    team_a_attack = sum(1 for x, y in curr_team_a_positions if attacking_zone[0] <= y < attacking_zone[1])
    team_b_defense = sum(1 for x, y in curr_team_b_positions if defensive_zone[0] <= y < defensive_zone[1])
    team_b_midfield = sum(1 for x, y in curr_team_b_positions if midfield_zone[0] <= y < midfield_zone[1])
    team_b_attack = sum(1 for x, y in curr_team_b_positions if attacking_zone[0] <= y < attacking_zone[1])

    # Calculate average distance between players for spacing comments
    def avg_player_distance(positions):
        if len(positions) < 2:
            return float('inf')
        total_dist = 0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                total_dist += dist
        return total_dist / (len(positions) * (len(positions) - 1) / 2)

    avg_dist_a = avg_player_distance(curr_team_a_positions)
    avg_dist_b = avg_player_distance(curr_team_b_positions)
    close_spacing_threshold = 50  # Pixels threshold for close proximity

    # Check player clustering (density in a zone)
    cluster_threshold = 4  # Minimum players for a cluster
    team_a_clustered = team_a_attack >= cluster_threshold or team_a_midfield >= cluster_threshold or team_a_defense >= cluster_threshold
    team_b_clustered = team_b_attack >= cluster_threshold or team_b_midfield >= cluster_threshold or team_b_defense >= cluster_threshold

    # Ball position and movement inference
    ball_y = detected_ball_dst_pos[1] if 'detected_ball_dst_pos' in locals() and detected_ball_dst_pos is not None else tac_height // 2
    ball_moving = len(ball_track_history['dst']) > 1 and np.linalg.norm(np.array(ball_track_history['dst'][-1]) - np.array(ball_track_history['dst'][-2])) > 50

    # Comment cooldown to avoid spam
    if 'last_comment_type' not in st.session_state:
        st.session_state['last_comment_type'] = None
        st.session_state['comment_cooldown'] = 0
    if st.session_state['comment_cooldown'] > 0:
        st.session_state['comment_cooldown'] -= 1

    # Ball and possession commentary
    if detected_ball_src_pos is not None and st.session_state['comment_cooldown'] == 0:
        if possession_team is not None:
            if ball_y > goal_area[0] and possession_team == team1_name:
                ball_comments = [
                    f"Frame {frame_nbr}: {team1_name} is in a dangerous spot near {team2_name}'s goal—what a chance!",
                    f"Frame {frame_nbr}: Ball whipped into the box by {team1_name}—dangerous play!",
                    f"Frame {frame_nbr}: {team1_name} is threatening the goal—keeper under pressure!"
                ]
                commentary.append(random.choice(ball_comments))
                st.session_state['comment_cooldown'] = 10
            elif ball_y > goal_area[0] and possession_team == team2_name:
                ball_comments = [
                    f"Frame {frame_nbr}: {team2_name} is threatening {team1_name}'s goal—tension is rising!",
                    f"Frame {frame_nbr}: Ball near the penalty area—{team2_name} smells a goal!",
                    f"Frame {frame_nbr}: {team2_name} launches an attack—goal in sight!"
                ]
                commentary.append(random.choice(ball_comments))
                st.session_state['comment_cooldown'] = 10
            elif ball_moving:
                movement_comments = [
                    f"Frame {frame_nbr}: The ball is being whipped across the field by {possession_team}!",
                    f"Frame {frame_nbr}: {possession_team} drives the ball with pace—exciting move!",
                    f"Frame {frame_nbr}: A swift pass by {possession_team}—the crowd is buzzing!"
                ]
                commentary.append(random.choice(movement_comments))
                st.session_state['comment_cooldown'] = 5
            else:
                zone_comments = {
                    attacking_zone: [
                        f"Frame {frame_nbr}: {possession_team} is pushing forward with intent!",
                        f"Frame {frame_nbr}: {possession_team} is on the attack—eyes on the goal!",
                        f"Frame {frame_nbr}: A striker is making a run for {possession_team}!"
                    ],
                    midfield_zone: [
                        f"Frame {frame_nbr}: {possession_team} is dictating the tempo in the middle!",
                        f"Frame {frame_nbr}: {possession_team} is holding the midfield with confidence!",
                        f"Frame {frame_nbr}: Midfield battle intensifies for {possession_team}!"
                    ],
                    defensive_zone: [
                        f"Frame {frame_nbr}: {possession_team} is keeping it tight in their own half!",
                        f"Frame {frame_nbr}: {possession_team} is playing it safe near their goal!",
                        f"Frame {frame_nbr}: A defender is holding the line for {possession_team}!"
                    ]
                }
                for zone, comments in zone_comments.items():
                    if zone[0] <= ball_y < zone[1]:
                        commentary.append(random.choice(comments))
                        st.session_state['comment_cooldown'] = 5
                        break
        else:
            ball_comments = [
                f"Frame {frame_nbr}: The ball is up for grabs—intense midfield battle!",
                f"Frame {frame_nbr}: Ball loose in the center—both teams scramble!",
                f"Frame {frame_nbr}: Neutral zone—possession is contested!"
            ]
            commentary.append(random.choice(ball_comments))
            st.session_state['comment_cooldown'] = 5
    elif detected_ball_src_pos is None and st.session_state['comment_cooldown'] == 0:
        ball_comments = [
            f"Frame {frame_nbr}: No ball in sight—momentary pause in the action!",
            f"Frame {frame_nbr}: Ball out of play—teams regrouping!",
            f"Frame {frame_nbr}: Brief stoppage—where’s the ball?"
        ]
        commentary.append(random.choice(ball_comments))
        st.session_state['comment_cooldown'] = 10

    # Player count and detection commentary
    if 'prev_player_count' in st.session_state and st.session_state['comment_cooldown'] == 0:
        curr_player_count = len([l for l in labels_p if l == 0])
        if curr_player_count != st.session_state['prev_player_count']:
            commentary.append(f"Frame {frame_nbr}: Player count drops to {curr_player_count}—has someone gone down?")
            st.session_state['prev_player_count'] = curr_player_count
            st.session_state['comment_cooldown'] = 10
    elif 'prev_player_count' not in st.session_state and st.session_state['comment_cooldown'] == 0:
        st.session_state['prev_player_count'] = len([l for l in labels_p if l == 0])
        commentary.append(f"Frame {frame_nbr}: Kickoff sees {st.session_state['prev_player_count']} players on the pitch!")
        st.session_state['comment_cooldown'] = 10

    # Tactical commentary with player-specific actions
    if st.session_state['comment_cooldown'] == 0:
        if team_a_attack > 2 and team_a_attack > team_b_defense and st.session_state['last_comment_type'] != 'attack':
            attack_adjectives = ["fierce", "relentless", "dynamic", "explosive"]
            player_comments = [
                f"Frame {frame_nbr}: {team1_name} is launching a {random.choice(attack_adjectives)} assault in the final third—the crowd roars!",
                f"Frame {frame_nbr}: {team1_name} striker leads a charge—goal attempt incoming!",
                f"Frame {frame_nbr}: {team1_name} players swarm the box—dangerous play!"
            ]
            commentary.append(random.choice(player_comments))
            st.session_state['last_comment_type'] = 'attack'
            st.session_state['comment_cooldown'] = 15
        elif team_b_attack > 2 and team_b_attack > team_a_defense and st.session_state['last_comment_type'] != 'attack':
            attack_adjectives = ["intense", "bold", "aggressive", "thunderous"]
            player_comments = [
                f"Frame {frame_nbr}: {team2_name} is mounting an {random.choice(attack_adjectives)} charge toward {team1_name}'s goal!",
                f"Frame {frame_nbr}: {team2_name} forward breaks through—shot on target?",
                f"Frame {frame_nbr}: {team2_name} attack builds momentum—keeper alert!"
            ]
            commentary.append(random.choice(player_comments))
            st.session_state['last_comment_type'] = 'attack'
            st.session_state['comment_cooldown'] = 15
        elif team_a_defense > 3 and team_a_midfield < 2 and st.session_state['last_comment_type'] != 'defense':
            defense_comments = [
                f"Frame {frame_nbr}: {team1_name} is building an impenetrable defensive wall—resolute stuff!",
                f"Frame {frame_nbr}: {team1_name} defenders hold firm—solid backline!",
                f"Frame {frame_nbr}: {team1_name} retreats to protect their goal!"
            ]
            commentary.append(random.choice(defense_comments))
            st.session_state['last_comment_type'] = 'defense'
            st.session_state['comment_cooldown'] = 15
        elif team_b_defense > 3 and team_b_midfield < 2 and st.session_state['last_comment_type'] != 'defense':
            defense_comments = [
                f"Frame {frame_nbr}: {team2_name} is retreating into a staunch defensive setup!",
                f"Frame {frame_nbr}: {team2_name} backline stands tall—great defending!",
                f"Frame {frame_nbr}: {team2_name} focuses on defense—tight formation!"
            ]
            commentary.append(random.choice(defense_comments))
            st.session_state['last_comment_type'] = 'defense'
            st.session_state['comment_cooldown'] = 15
        elif len(team_a_positions) > len(team_b_positions) * 1.5 and possession_team == team1_name and st.session_state['last_comment_type'] != 'dominance':
            commentary.append(f"Frame {frame_nbr}: {team1_name} is asserting total dominance with a numerical advantage—impressive!")
            st.session_state['last_comment_type'] = 'dominance'
            st.session_state['comment_cooldown'] = 15
        elif len(team_b_positions) > len(team_a_positions) * 1.5 and possession_team == team2_name and st.session_state['last_comment_type'] != 'dominance':
            commentary.append(f"Frame {frame_nbr}: {team2_name} is taking the upper hand with superior numbers—what a shift!")
            st.session_state['last_comment_type'] = 'dominance'
            st.session_state['comment_cooldown'] = 15

    # Player coordination and spacing commentary
    if st.session_state['comment_cooldown'] == 0:
        if team_a_clustered and st.session_state['last_comment_type'] != 'coordination':
            zone = max([(team_a_attack, attacking_zone), (team_a_midfield, midfield_zone), (team_a_defense, defensive_zone)], key=lambda x: x[0])[1]
            coordination_comments = {
                attacking_zone: [
                    f"Frame {frame_nbr}: {team1_name} is forming a tight attacking unit in the final third!",
                    f"Frame {frame_nbr}: {team1_name} players cluster near the goal—crowd roars!",
                    f"Frame {frame_nbr}: {team1_name} attack builds with close coordination!"
                ],
                midfield_zone: [
                    f"Frame {frame_nbr}: {team1_name} is linking up brilliantly in the midfield!",
                    f"Frame {frame_nbr}: {team1_name} midfielders create a solid wall!",
                    f"Frame {frame_nbr}: {team1_name} controls the center with teamwork!"
                ],
                defensive_zone: [
                    f"Frame {frame_nbr}: {team1_name} is building a solid defensive line!",
                    f"Frame {frame_nbr}: {team1_name} defenders pack the back—strong stance!",
                    f"Frame {frame_nbr}: {team1_name} holds a tight defensive shape!"
                ]
            }
            commentary.append(random.choice(coordination_comments[zone]))
            st.session_state['last_comment_type'] = 'coordination'
            st.session_state['comment_cooldown'] = 15
        elif team_b_clustered and st.session_state['last_comment_type'] != 'coordination':
            zone = max([(team_b_attack, attacking_zone), (team_b_midfield, midfield_zone), (team_b_defense, defensive_zone)], key=lambda x: x[0])[1]
            coordination_comments = {
                attacking_zone: [
                    f"Frame {frame_nbr}: {team2_name} is coordinating a strong push upfield!",
                    f"Frame {frame_nbr}: {team2_name} players swarm the attack—intense pressure!",
                    f"Frame {frame_nbr}: {team2_name} builds an offensive cluster!"
                ],
                midfield_zone: [
                    f"Frame {frame_nbr}: {team2_name} is showing great teamwork in the midfield!",
                    f"Frame {frame_nbr}: {team2_name} midfield holds strong—great passing!",
                    f"Frame {frame_nbr}: {team2_name} dominates the center with unity!"
                ],
                defensive_zone: [
                    f"Frame {frame_nbr}: {team2_name} is setting up a formidable defensive block!",
                    f"Frame {frame_nbr}: {team2_name} backline tightens up—solid defense!",
                    f"Frame {frame_nbr}: {team2_name} forms a defensive fortress!"
                ]
            }
            commentary.append(random.choice(coordination_comments[zone]))
            st.session_state['last_comment_type'] = 'coordination'
            st.session_state['comment_cooldown'] = 15

        # Distance-based comments
        if avg_dist_a < close_spacing_threshold and st.session_state['last_comment_type'] != 'spacing':
            commentary.append(f"Frame {frame_nbr}: {team1_name} players are tightly packed—risk of a collision!")
            st.session_state['last_comment_type'] = 'spacing'
            st.session_state['comment_cooldown'] = 15
        elif avg_dist_a > 100 and st.session_state['last_comment_type'] != 'spacing':
            commentary.append(f"Frame {frame_nbr}: {team1_name} players are spreading out—good spacing!")
            st.session_state['last_comment_type'] = 'spacing'
            st.session_state['comment_cooldown'] = 15
        elif avg_dist_b < close_spacing_threshold and st.session_state['last_comment_type'] != 'spacing':
            commentary.append(f"Frame {frame_nbr}: {team2_name} players are closely bunched—tactical move or mistake?")
            st.session_state['last_comment_type'] = 'spacing'
            st.session_state['comment_cooldown'] = 15
        elif avg_dist_b > 100 and st.session_state['last_comment_type'] != 'spacing':
            commentary.append(f"Frame {frame_nbr}: {team2_name} players are well-spread—open play developing!")
            st.session_state['last_comment_type'] = 'spacing'
            st.session_state['comment_cooldown'] = 15

    # Game momentum and late-game pressure
    if frame_nbr > tot_nbr_frames * 0.75 and st.session_state['comment_cooldown'] == 0 and st.session_state['last_comment_type'] != 'momentum':
        momentum_comments = [
            f"Frame {frame_nbr}: {team1_name} is piling on the pressure in the final stages—can they score?",
            f"Frame {frame_nbr}: {team2_name} is making a late surge—game on!",
            f"Frame {frame_nbr}: Final minutes—both teams push for a goal!",
            f"Frame {frame_nbr}: Late-game tension rises—{team1_name} looks desperate!"
        ]
        commentary.append(random.choice(momentum_comments))
        st.session_state['last_comment_type'] = 'momentum'
        st.session_state['comment_cooldown'] = 20

    # Referee and foul commentary (random low chance)
    if random.random() < 0.05 and st.session_state['comment_cooldown'] == 0 and st.session_state['last_comment_type'] != 'referee':
        referee_comments = [
            f"Frame {frame_nbr}: Was that a foul? The referee is under scrutiny—tense moment!",
            f"Frame {frame_nbr}: Referee checks for offside—drama unfolds!",
            f"Frame {frame_nbr}: Possible handball—referee deliberates!",
            f"Frame {frame_nbr}: Yellow card waved—{random.choice([team1_name, team2_name])} in trouble!"
        ]
        commentary.append(random.choice(referee_comments))
        st.session_state['last_comment_type'] = 'referee'
        st.session_state['comment_cooldown'] = 20

    # Crowd reactions and near misses
    if random.random() < 0.1 and st.session_state['comment_cooldown'] == 0 and st.session_state['last_comment_type'] != 'reaction':
        if ball_y > goal_area[0] and possession_team:
            reaction_comments = [
                f"Frame {frame_nbr}: That was close—a near miss for {possession_team}! The crowd holds its breath!",
                f"Frame {frame_nbr}: Fans gasp as the shot goes wide for {possession_team}!",
                f"Frame {frame_nbr}: {possession_team} strikes—off the post, what a save!"
            ]
            commentary.append(random.choice(reaction_comments))
        else:
            crowd_reactions = [
                f"Frame {frame_nbr}: The crowd erupts as {team1_name} takes control!",
                f"Frame {frame_nbr}: Fans are on their feet—{team2_name} is fighting back!",
                f"Frame {frame_nbr}: A deafening roar as the tension builds!",
                f"Frame {frame_nbr}: Crowd cheers a brilliant tackle by {random.choice([team1_name, team2_name])}!"
            ]
            commentary.append(random.choice(crowd_reactions))
        st.session_state['last_comment_type'] = 'reaction'
        st.session_state['comment_cooldown'] = 15

    # Reset cooldown if no new comment
    if not commentary and st.session_state['comment_cooldown'] > 0:
        st.session_state['comment_cooldown'] = max(0, st.session_state['comment_cooldown'] - 1)
    elif not commentary:
        st.session_state['last_comment_type'] = None

    return commentary

# Update create_heatmap to include zone and flank analysis (from second detection.py)
def create_heatmap(positions, base_map, output_path):
    # Create a blank heatmap array
    heatmap_data = np.zeros((base_map.shape[0], base_map.shape[1]))
    
    # Add points to heatmap
    for x, y in positions:
        if 0 <= x < heatmap_data.shape[1] and 0 <= y < heatmap_data.shape[0]:
            heatmap_data[y, x] += 1
    
    # Apply Gaussian blur to smooth the heatmap
    heatmap_data = gaussian_filter(heatmap_data, sigma=20)
    
    # Normalize heatmap data
    if heatmap_data.max() > 0:
        heatmap_data = heatmap_data / heatmap_data.max()
    
    # Divide heatmap into zones for analysis (adjusted for bottom-attacking orientation)
    tac_height, tac_width = base_map.shape[:2]
    defensive_zone = slice(0, tac_height // 3)  # Top third (defense)
    midfield_zone = slice(tac_height // 3, 2 * tac_height // 3)  # Middle third
    attacking_zone = slice(2 * tac_height // 3, tac_height)  # Bottom third (attack)
    left_flank = slice(0, tac_width // 2)
    right_flank = slice(tac_width // 2, tac_width)

    # Compute zone and flank intensities (using sum for more sensitivity)
    zone_intensities = {
        'defensive': np.sum(heatmap_data[defensive_zone]),
        'midfield': np.sum(heatmap_data[midfield_zone]),
        'attacking': np.sum(heatmap_data[attacking_zone])
    }
    flank_intensities = {
        'left': np.sum(heatmap_data[:, left_flank]),
        'right': np.sum(heatmap_data[:, right_flank])
    }

    # Create heatmap visualization
    plt.figure(figsize=(base_map.shape[1]/100, base_map.shape[0]/100))
    plt.imshow(cv2.cvtColor(base_map, cv2.COLOR_BGR2RGB))
    plt.imshow(heatmap_data, cmap='hot', alpha=0.6, interpolation='nearest')
    plt.axis('off')
    
    # Save the heatmap
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Load the saved heatmap as an image
    heatmap_img = cv2.imread(output_path)
    return heatmap_img, zone_intensities, flank_intensities

def detect(cap, stframe, heatmap1, heatmap2, trace1, trace2, commentary_placeholder, output_file_name, save_output, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab):
    global labels_p, tot_nbr_frames  # Add global variables as in second detection.py

    # Initialize variables for heatmap
    team_a_positions = []
    team_b_positions = []
    team_names = list(colors_dic.keys())

    show_k = plot_hyperparams[0]
    show_pal = plot_hyperparams[1]
    show_b = plot_hyperparams[2]
    show_p = plot_hyperparams[3]
    show_possession = plot_hyperparams[4]  # Add possession toggle

    p_conf = hyper_params[0]
    k_conf = hyper_params[1]
    k_d_tol = hyper_params[2]

    nbr_frames_no_ball_thresh = ball_track_hyperparams[0]
    ball_track_dist_thresh = ball_track_hyperparams[1]
    max_track_length = ball_track_hyperparams[2]

    nbr_team_colors = len(list(colors_dic.values())[0])

    if (output_file_name is not None) and (len(output_file_name) == 0):
        output_file_name = generate_file_name()

    # Read tactical map image
    tac_map = cv2.imread('../tactical map.jpg')
    tac_width = tac_map.shape[0]
    tac_height = tac_map.shape[1]
    
    # Create output video writer
    if save_output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
        output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 45.0, (width, height))

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text='Detection starting.')

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history in a dictionary
    ball_track_history = {'src': [], 'dst': []}

    nbr_frames_no_ball = 0

    tac_map_copy_team_a = tac_map.copy()
    tac_map_copy_team_b = tac_map.copy()
    team_chooser = ""

    # New variable for ball possession
    possession_team = None
    possession_threshold = 100

    # Add variables for commentary
    commentary_log = []
    detected_labels_prev = []
    detected_labels_src_pts_prev = []

    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames + 1):
        # Update progress bar
        percent_complete = int(frame_nbr / (tot_nbr_frames) * 100)
        st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()
        tac_map_1 = tac_map.copy()

        if nbr_frames_no_ball > nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:
            #################### Part 1 ####################
            # Object Detection & Coordinate Transformation #
            ################################################

            # Run YOLOv8 players inference on the frame
            results_players = model_players(frame, conf=p_conf)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=k_conf)

            ## Extract detections information
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordinates on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])

            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                if frame_nbr > 1:
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)
                        update_homography = coor_error > k_d_tol
                    else:
                        update_homography = True
                else:
                    update_homography = True

                if update_homography:
                    homog, mask = cv2.findHomography(detected_labels_src_pts, detected_labels_dst_pts)
            
            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy()
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()

                bboxes_p_c_0 = bboxes_p_c[[i == 0 for i in labels_p], :]
                bboxes_p_c_2 = bboxes_p_c[[i == 2 for i in labels_p], :]

                # Get coordinates of detected players on frame (x_center, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:, :2] + np.array([[0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3] / 2]).transpose()
                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[0, :2] if bboxes_p_c_2.shape[0] > 0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball += 1
                    possession_team = None
                else:
                    nbr_frames_no_ball = 0

                # Transform players coordinates from frame plane to tactical map plane using the calculated Homography matrix
                pred_dst_pts = []
                for pt in detected_ppos_src_pts:
                    pt = np.append(np.array(pt), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point / dest_point[2]
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))
                pred_dst_pts = np.array(pred_dst_pts)

                # Transform ball coordinates from frame plane to tactical map plane using the calculated Homography matrix
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point / dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # Track ball history
                    if show_b:
                        if len(ball_track_history['src']) > 0:
                            if np.linalg.norm(detected_ball_src_pos - ball_track_history['src'][-1]) < ball_track_dist_thresh:
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                ball_track_history['src'] = [(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst'] = [(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                
                if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            obj_palette_list = []
            palette_interval = (0, num_pal_colors)

            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i, :]
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w // 2) - (obj_img_w // 5), 1])
                    center_filter_x2 = (obj_img_w // 2) + (obj_img_w // 5)
                    center_filter_y1 = np.max([(obj_img_h // 3) - (obj_img_h // 5), 1])
                    center_filter_y2 = (obj_img_h // 3) + (obj_img_h // 5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)
                    palette = reduced.getpalette()
                    palette = [palette[3 * n:3 * n + 3] for n in range(256)]
                    color_count = [(n, palette[m]) for n, m in reduced.getcolors()]
                    RGB_df = pd.DataFrame(color_count, columns=['cnt', 'RGB']).sort_values(
                                        by='cnt', ascending=False).iloc[
                                            palette_interval[0]:palette_interval[1], :]
                    palette = list(RGB_df.RGB)
                    obj_palette_list.append(palette)

            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]
                for color in palette_lab:
                    distance_list = []
                    for c in color_list_lab:
                        distance = skimage.color.deltaE_cie76(color, c)
                        distance_list.append(distance)
                    palette_distance.append(distance_list)
                players_distance_features.append(palette_distance)

            ## Predict detected players teams based on distance features
            players_teams_list = []
            for distance_feats in players_distance_features:
                vote_list = []
                for dist_list in distance_feats:
                    team_idx = dist_list.index(min(dist_list)) // nbr_team_colors
                    vote_list.append(team_idx)
                players_teams_list.append(max(vote_list, key=vote_list.count))

            # Ball Possession Logic
            if detected_ball_src_pos is not None and len(detected_ppos_src_pts) > 0:
                distances = [np.linalg.norm(detected_ball_src_pos - player_pos) for player_pos in detected_ppos_src_pts]
                min_distance_idx = np.argmin(distances)
                if distances[min_distance_idx] < possession_threshold:
                    possession_team = list(colors_dic.keys())[players_teams_list[min_distance_idx]]
                else:
                    possession_team = None

            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################

            ball_color_bgr = (0, 0, 255)
            j = 0
            k = 1
            palette_box_size = 10
            annotated_frame = frame

            # Loop over all detected objects by players detection model
            for i in range(bboxes_p.shape[0]):
                conf = confs_p[i]
                if labels_p[i] == 0:
                    if 'homog' in locals():
                        team_name = list(colors_dic.keys())[players_teams_list[j]]
                        player_pos = (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1]))
                        if team_name == team_names[0]:
                            team_a_positions.append(player_pos)
                        elif team_name == team_names[1]:
                            team_b_positions.append(player_pos)

                    if show_pal:
                        palette = obj_palette_list[j]
                        for k, c in enumerate(palette):
                            c_bgr = c[::-1]
                            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 2]) + 3,
                                                                    int(bboxes_p[i, 1]) + k * palette_box_size),
                                                                    (int(bboxes_p[i, 2]) + palette_box_size,
                                                                    int(bboxes_p[i, 1]) + (palette_box_size) * (k + 1)),
                                                                    c_bgr, -1)
                    team_name = list(colors_dic.keys())[players_teams_list[j]]
                    color_rgb = colors_dic[team_name][0]
                    color_bgr = color_rgb[::-1]
                    if show_p:
                        annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])),
                                                        (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), color_bgr, 1)
                        annotated_frame = cv2.putText(annotated_frame, team_name + f" {conf:.2f}",
                                    (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_bgr, 2)

                    if 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                                            radius=5, color=color_bgr, thickness=-1)
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                                            radius=5, color=(0, 0, 0), thickness=1)
                        if team_name == next(iter(colors_dic)):
                            tac_map_1 = tac_map_copy_team_a
                        elif team_name == list(colors_dic)[1]:
                            tac_map_1 = tac_map_copy_team_b
                        tac_map_1 = cv2.circle(tac_map_1, (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                                            radius=2, color=color_bgr, thickness=1)
                    if team_name == next(iter(colors_dic)):
                        tac_map_copy_team_a = tac_map_1
                    elif team_name == list(colors_dic)[1]:
                        tac_map_copy_team_b = tac_map_1
                    tac_map_1 = tac_map.copy()
                    j += 1
                    k = j + 1
                else:
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i, 0]), int(bboxes_p[i, 1])),
                                                    (int(bboxes_p[i, 2]), int(bboxes_p[i, 3])), (255, 255, 255), 1)
                    annotated_frame = cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",
                                (int(bboxes_p[i, 0]), int(bboxes_p[i, 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 2)

                    if detected_ball_src_pos is not None and 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]), 
                                                    int(detected_ball_dst_pos[1])), radius=5, 
                                                    color=ball_color_bgr, thickness=3)
            
            if show_k:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i, 0]), int(bboxes_k[i, 1])),
                                                (int(bboxes_k[i, 2]), int(bboxes_k[i, 3])), (0, 0, 0), 1)

            # Plot the tracks
            if len(ball_track_history['src']) > 0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)

            # Add Ball Possession Annotation
            if show_possession and possession_team is not None:
                possession_color = colors_dic[possession_team][0][::-1]
                cv2.putText(annotated_frame, f"Possession: {possession_team}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, possession_color, 2)
            elif show_possession:
                cv2.putText(annotated_frame, "Possession: None", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Combine annotated frame and tactical map in one image with colored border separation
            border_color = [255, 255, 255]
            annotated_frame = cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))
            final_img = cv2.hconcat((annotated_frame, tac_map_copy))

            ## Add info annotation
            cv2.putText(final_img, "Tactical Map", (1370, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

            # Display the annotated frame
            stframe.image(final_img, channels="BGR")
            if save_output:
                output.write(cv2.resize(final_img, (width, height)))

            # Generate and display commentary in real-time
            commentary = generate_commentary(frame_nbr, possession_team, detected_ball_src_pos, team_names, team_a_positions, team_b_positions, tac_height, tac_width, ball_track_history, labels_p)
            commentary_log.extend(commentary)
            # Update the commentary placeholder with the latest log
            with commentary_placeholder.container():
                st.write("### Analyst Commentary Log")
                for comment in commentary_log:
                    st.write(comment)

    # After the loop, generate and save heatmaps with analysis
    print(f"Team A Positions: {len(team_a_positions)} points")
    print(f"Team B Positions: {len(team_b_positions)} points")
    
    # Generate and save heatmaps for both teams with zone and flank intensities
    heatmap_team_a, zone_intensities_a, flank_intensities_a = create_heatmap(team_a_positions, tac_map, './outputs/heatmap_team_A.jpg')
    heatmap_team_b, zone_intensities_b, flank_intensities_b = create_heatmap(team_b_positions, tac_map, './outputs/heatmap_team_B.jpg')
    print(f"Team A Zones: {zone_intensities_a}")
    print(f"Team B Zones: {zone_intensities_b}")
    print(f"Team A Flanks: {flank_intensities_a}")
    print(f"Team B Flanks: {flank_intensities_b}")

    # Save trace images
    cv2.imwrite(os.path.join("./outputs", "output_image_team_A.jpg"), tac_map_copy_team_a)
    cv2.imwrite(os.path.join("./outputs", "output_image_team_B.jpg"), tac_map_copy_team_b)

    # Update Streamlit display and store images in session state
    heatmap1.image(heatmap_team_a, channels="BGR")
    heatmap2.image(heatmap_team_b, channels="BGR")
    trace1.image(tac_map_copy_team_a, channels="BGR")
    trace2.image(tac_map_copy_team_b, channels="BGR")
    
    # Store images in session state for persistence
    st.session_state['heatmap1_image'] = heatmap_team_a
    st.session_state['heatmap2_image'] = heatmap_team_b
    st.session_state['trace1_image'] = tac_map_copy_team_a
    st.session_state['trace2_image'] = tac_map_copy_team_b

    # Debug: Print commentary log before analysis
    print("Commentary Log Before Analysis:", commentary_log)

    # Detailed post-match heatmap analysis (from second detection.py)
    commentary_log.append("Post-Match Analysis: Heatmap Insights")
    team1_name, team2_name = team_names

    # Relaxed thresholds to 1.05x for zones and 1.02x for flanks
    if zone_intensities_a['attacking'] > zone_intensities_b['attacking'] * 1.05:
        commentary_log.append(f"- {team1_name} dominated the attacking third with aggressive play!")
    elif zone_intensities_b['attacking'] > zone_intensities_a['attacking'] * 1.05:
        commentary_log.append(f"- {team2_name} controlled the attacking third with relentless pressure!")

    if zone_intensities_a['midfield'] > zone_intensities_b['midfield'] * 1.05:
        commentary_log.append(f"- {team1_name} held a strong grip on the midfield throughout the match!")
    elif zone_intensities_b['midfield'] > zone_intensities_a['midfield'] * 1.05:
        commentary_log.append(f"- {team2_name} dictated the midfield with superior coordination!")

    if zone_intensities_a['defensive'] > zone_intensities_b['defensive'] * 1.05:
        commentary_log.append(f"- {team1_name} showcased a solid defensive stance in their own half!")
    elif zone_intensities_b['defensive'] > zone_intensities_a['defensive'] * 1.05:
        commentary_log.append(f"- {team2_name} maintained a robust defense in their territory!")

    # Flank dominance (even more relaxed threshold to 1.02x)
    if flank_intensities_a['left'] > flank_intensities_a['right'] * 1.02:
        commentary_log.append(f"- {team1_name} dominated the left flank with effective wing play!")
    elif flank_intensities_a['right'] > flank_intensities_a['left'] * 1.02:
        commentary_log.append(f"- {team1_name} excelled on the right flank with dynamic movements!")

    if flank_intensities_b['left'] > flank_intensities_b['right'] * 1.02:
        commentary_log.append(f"- {team2_name} controlled the left flank with strong positioning!")
    elif flank_intensities_b['right'] > flank_intensities_b['left'] * 1.02:
        commentary_log.append(f"- {team2_name} showcased dominance on the right flank!")

    # Overall match summary (relaxed threshold to 1.05x)
    overall_intensity_a = np.mean(list(zone_intensities_a.values()))
    overall_intensity_b = np.mean(list(zone_intensities_b.values()))
    if overall_intensity_a > overall_intensity_b * 1.05:
        commentary_log.append(f"Final Verdict: {team1_name} dominated the match with intense activity across the pitch!")
    elif overall_intensity_b > overall_intensity_a * 1.05:
        commentary_log.append(f"Final Verdict: {team2_name} showcased superior control and energy throughout!")
    else:
        commentary_log.append(f"Final Verdict: A closely contested battle—both {team1_name} and {team2_name} gave their all!")

    # Fallback analysis if no zone/flank comments are generated
    if not any(comment.startswith("- ") for comment in commentary_log):
        max_zone_a = max(zone_intensities_a, key=zone_intensities_a.get)
        max_zone_b = max(zone_intensities_b, key=zone_intensities_b.get)
        commentary_log.append(f"- {team1_name} was most active in the {max_zone_a} third!")
        commentary_log.append(f"- {team2_name} focused heavily in the {max_zone_b} third!")
        commentary_log.append(f"Final Verdict: The match featured distinct tactical approaches from both teams!")

    # Debug: Print commentary log after analysis
    print("Commentary Log After Analysis:", commentary_log)

    # Save commentary to file and update session state
    with open('./outputs/commentary.txt', 'w') as f:
        for line in commentary_log:
            f.write(line + '\n')
    st.session_state['commentary'] = commentary_log
    st.session_state['heatmap_analysis'] = [comment for comment in commentary_log if comment.startswith("- ") or comment.startswith("Final Verdict:")]

    # Remove progress bar and return
    st_prog_bar.empty()
    return True