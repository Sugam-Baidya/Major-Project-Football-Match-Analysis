# Import libraries
import numpy as np # powerful library for numerical computing, supporting arrays, matrices, and mathematical functions.
import pandas as pd #  Used for data manipulation and analysis, particularly with tabular data like CSV files.
import streamlit as st #  Python framework for building interactive web applications


import cv2 # (OpenCV) popular library for image processing and computer vision tasks (e.g., reading images, edge detection, transformations).
import skimage # (scikit-image) – Another image processing library, often used for advanced filtering, segmentation, and transformations.
from PIL import Image, ImageColor # Used for working with images (e.g., opening, editing, and converting formats).
from ultralytics import YOLO #import model 
from sklearn.metrics import mean_squared_error #  Used to calculate the Mean Squared Error (MSE), a metric for evaluating the performance of regression models.

import os # Provides functions to interact with the operating system
import json # Handles JSON (JavaScript Object Notation) data, useful for reading and writing structured data.
import yaml # used to read, write, and parse YAML (YAML Ain't Markup Language) files.
import time # Provides time-related functions (e.g., delays, timestamps).

import matplotlib.pyplot as plt                    # For visualization
from matplotlib.colors import LinearSegmentedColormap  # For custom color maps
from scipy.ndimage import gaussian_filter

def get_labels_dics():
    ## Get tactical map keypoints positions dictionary ##
    json_path = "../pitch map labels position.json"
    with open(json_path, 'r') as f: # opens a file located at json_path in read mode ('r') and assigns it to variable f
        keypoints_map_pos = json.load(f) #reads th content of open file "f" and converts it into python list/dictionary and stores in keypoints_map_pos

    ## Get football field keypoints numerical to alphabetical mapping ##
    yaml_path = "../config pitch dataset.yaml"
    with open(yaml_path, 'r') as file:
        classes_names_dic = yaml.safe_load(file) # reads and parses the YAML file into a Python dictionary ; safe_load() ensures security by preventing code execution
    classes_names_dic = classes_names_dic['names'] # extracts and stores only the value associated with "names" into classes_names_dic.

    ## Get football field players numerical to alphabetical mapping ##
    yaml_path = "../config players dataset.yaml"
    with open(yaml_path, 'r') as file:
        labels_dic = yaml.safe_load(file)
    labels_dic = labels_dic['names']
    return keypoints_map_pos, classes_names_dic, labels_dic # functions returns keypoint position, field class names mapped with their numbers and player,ball,rfree mapped to numbers

def create_colors_info(team1_name, team1_p_color, team1_gk_color, team2_name, team2_p_color, team2_gk_color):
    # ImageColor.getcolor(color, "RGB") is a function from the Pillow (PIL) library that converts a color name or hex code into an RGB tuple.
    # Converting colours in HEX code into RGB format
    team1_p_color_rgb = ImageColor.getcolor(team1_p_color, "RGB") 
    team1_gk_color_rgb = ImageColor.getcolor(team1_gk_color, "RGB")
    team2_p_color_rgb = ImageColor.getcolor(team2_p_color, "RGB")
    team2_gk_color_rgb = ImageColor.getcolor(team2_gk_color, "RGB")

    # create a dictionary to map teams to their colors(both player and gk)
    colors_dic = {
        team1_name:[team1_p_color_rgb, team1_gk_color_rgb],
        team2_name:[team2_p_color_rgb, team2_gk_color_rgb]
    }

    colors_list = colors_dic[team1_name]+colors_dic[team2_name] # Define color list to be used for detected player team prediction in RGB format
    # yesto format hunxa [[(x,y,z),(x,y,z)],[(x,y,z),(x,y,z)]]
    color_list_lab = [skimage.color.rgb2lab([i/255 for i in c]) for c in colors_list] # Converting color_list to L*a*b* space ;i/255 for i in c normalizes the RGB values from (0-255) to (0-1).
    return colors_dic, color_list_lab # returns colour dictionary and color list in lab format

# yo def cahi output file banauna lai 
def generate_file_name():
    list_video_files = os.listdir('./outputs/') # accessing 'os' module; retrieves a list of all files in the ./outputs/ directory and stores them in the list_video_files list.
    idx = 0 #used to generate file name
    while True:
        idx +=1
        output_file_name = f'detect_{idx}'
        if output_file_name+'.mp4' not in list_video_files:
            break
    return output_file_name

def detect(cap, stframe,heatmap1,heatmap2,trace1,trace2, output_file_name, save_output, model_players, model_keypoints,
            hyper_params, ball_track_hyperparams, plot_hyperparams, num_pal_colors, colors_dic, color_list_lab):
    # cap:- Likely an OpenCV video capture object (cv2.VideoCapture()) used to process video frames.
    # stframe:- Appears to be related to Streamlit, meaning the function may be displaying real-time detection output in a Streamlit app.
    
    # Initialize variables for heatmap
    team_a_positions = []
    team_b_positions = []
    team_names = list(colors_dic.keys())

    show_k = plot_hyperparams[0]
    show_pal = plot_hyperparams[1]
    show_b = plot_hyperparams[2]
    show_p = plot_hyperparams[3]

    p_conf = hyper_params[0]
    k_conf = hyper_params[1]
    k_d_tol = hyper_params[2]

    nbr_frames_no_ball_thresh = ball_track_hyperparams[0]
    ball_track_dist_thresh = ball_track_hyperparams[1]
    max_track_length = ball_track_hyperparams[2]

    nbr_team_colors = len(list(colors_dic.values())[0])

    if (output_file_name is not None) and (len(output_file_name)==0):
        output_file_name = generate_file_name()

    # Read tactical map image
    tac_map = cv2.imread('../tactical map.jpg')  # read tactical_map.jpg and store it in tac_map;in array (height, width, channels)
    tac_map_1 = cv2.imread('../tactical map.jpg')
    tac_width = tac_map.shape[0] #  extracts the first dimension of the shape, which represents the height of the array; corresponds to the number of pixels in the vertical direction.
    tac_height = tac_map.shape[1] # extracts the second dimension of the shape, which represents the width of the array; corresponds to the number of pixels in the horizontal direction.
    
    # Create output video writer; output video banauxa yesle original video ra tactical map ko size jodera
    if save_output: # checks if the save output flag is true
        # cap.get(cv2.CAP_PROP_FRAME_WIDTH) retrieves the width of the frames from the video capture object (cap)/video; adding tac wieght
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + tac_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + tac_height
        output = cv2.VideoWriter(f'./outputs/{output_file_name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 45.0, (width, height)) # *mp4v is codex

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # retrieves the total number of frames in the video file or stream and int() converts it into integer
    st_prog_bar = st.progress(0, text='Detection starting.') # 0% Detection Strating bhanera euta progress bar banauxa

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Set variable to record the time when we processed last frame 
    prev_frame_time = 0
    # Set variable to record the time at which we processed current frame 
    new_frame_time = 0
    
    # Store the ball track history in a dictionary, 'src' and 'dst' are keyvalue pair linked to a list
    ball_track_history = {'src':[], #source
                          'dst':[] #destination
    }

    nbr_frames_no_ball = 0

    tac_map_copy_team_a = tac_map.copy()
    tac_map_copy_team_b = tac_map.copy()
    team_chooser = ""

    # New variable for ball possession
    possession_team = None  # Will store the team name with possession
    possession_threshold = 100  # Distance threshold (in pixels) to determine possession

    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames+1): # frame_nbr: current frame number in the loop

        # Update progress bar
        # percentage nikalera update garne progress bar. Percentage cahi total no of frames ra processded frames bata nikalne
        percent_complete = int(frame_nbr/(tot_nbr_frames)*100)
        st_prog_bar.progress(percent_complete, text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        #cap.read() returns two values 
        # frame = The actual frame data, which is typically a NumPy array representing the image
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        # Modifications to tac_map_copy won't affect the original tac_map
        # If you don't use .copy(), and you just do tac_map_copy = tac_map, both variables will point to the same underlying data. Any change made to one will be reflected in the other.

        tac_map_copy = tac_map.copy() 
        #tac_map_1 = tac_map.copy()
        #if team_chooser != "":
            #if team_chooser == "a":
                #tac_map_1 = tac_map_copy_team_a.copy() 
           # elif team_chooser == "b":
                #tac_map_1 = tac_map_copy_team_b.copy()

        if nbr_frames_no_ball>nbr_frames_no_ball_thresh:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:

            #################### Part 1 ####################
            # Object Detection & Coordiante Transofrmation #
            ################################################

            # Run YOLOv8 players inference on the frame
            # Results_players is likely an object that contains the model's output, which typically includes:
            # Bounding boxes for the detected objects (players).
            # Class labels for the detected objects.
            # Confidence scores for each detection.
            results_players = model_players(frame, conf=p_conf)
            # Run YOLOv8 field keypoints inference on the frame
            results_keypoints = model_keypoints(frame, conf=k_conf)
            
            

            ## Extract detections information
            # .cpu : moves the tensor (if it's stored on a GPU) to the CPU
            bboxes_p = results_players[0].boxes.xyxy.cpu().numpy()                          # Detected players, referees and ball (x,y,x,y) bounding boxes; coordinates f the bounding box
            bboxes_p_c = results_players[0].boxes.xywh.cpu().numpy()                        # Detected players, referees and ball (x,y,w,h) bounding boxes; coordinates of center and width and height of box
            labels_p = list(results_players[0].boxes.cls.cpu().numpy())                     # Detected players, referees and ball labels list
            confs_p = list(results_players[0].boxes.conf.cpu().numpy())                     # Detected players, referees and ball confidence level
            
            bboxes_k = results_keypoints[0].boxes.xyxy.cpu().numpy()                        # Detected field keypoints (x,y,x,y) bounding boxes
            bboxes_k_c = results_keypoints[0].boxes.xywh.cpu().numpy()                      # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(results_keypoints[0].boxes.cls.cpu().numpy())                   # Detected field keypoints labels list

            

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([list(np.round(bboxes_k_c[i][:2]).astype(int)) for i in range(bboxes_k_c.shape[0])])
            #[:2], you are selecting only the first two elements of the bounding box — the x and y coordinates of the center.
            # np.round() rounds the x and y coordinates to the nearest integer.
            # .astype(int):converts the rounded coordinates from float to integer type.
            # gives total number of bounding boxes detected.

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array([keypoints_map_pos[i] for i in detected_labels])


            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(detected_labels)
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [detected_labels_prev.index(i) for i in common_labels]   # Get labels indexes of common detected keypoints from previous frame
                        common_label_idx_curr = [detected_labels.index(i) for i in common_labels]        # Get labels indexes of common detected keypoints from current frame
                        coor_common_label_prev = detected_labels_src_pts_prev[common_label_idx_prev]     # Get labels coordiantes of common detected keypoints from previous frame
                        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]          # Get labels coordiantes of common detected keypoints from current frame
                        coor_error = mean_squared_error(coor_common_label_prev, coor_common_label_curr)  # Calculate error between previous and current common keypoints coordinates
                        update_homography = coor_error > k_d_tol                                         # Check if error surpassed the predefined tolerance level
                    else:
                        update_homography = True                                                         
                else:
                    update_homography = True

                if  update_homography:
                    homog, mask = cv2.findHomography(detected_labels_src_pts,                  # Calculate homography matrix using openCVs findHomography() function
                                                detected_labels_dst_pts)                  
            if 'homog' in locals():                                                         # checks whether the variable homog(3*3 matrix) exists in the current local scope.
                detected_labels_prev = detected_labels.copy()                               # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy()               # Save current detected keypoint coordiantes for next frame

                bboxes_p_c_0 = bboxes_p_c[[i==0 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected players (label 0)
                bboxes_p_c_2 = bboxes_p_c[[i==2 for i in labels_p],:]                       # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

                # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:,:2]  + np.array([[0]*bboxes_p_c_0.shape[0], bboxes_p_c_0[:,3]/2]).transpose()
                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[0,:2] if bboxes_p_c_2.shape[0]>0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball+=1
                    possession_team = None  # No possession if no ball is detected
                else: 
                    nbr_frames_no_ball=0

                # Transform players coordinates from frame plane to tactical map plance using the calculated Homography matrix
                pred_dst_pts = []                                                           # Initialize players tactical map coordinates list;stores transformed player positions on tactical map
                for pt in detected_ppos_src_pts:                                            # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]), axis=0)                     # Covert to homogeneous coordiantes
                    dest_point = np.matmul(homog, np.transpose(pt))                         # Apply homography transofrmation
                    #dest_point is a numpy array.
                    #suru ma pt bhaneko (x,y) from video coordinate
                    #yeslai maltrix multiply garna (x,y,1) banaunu parx which is called homogeneous coordinates
                    #matrix multiplicaion garne homoraphy apply garna
                    # ani divide by 3rd element garn to remove the third added component 
                    dest_point = dest_point/dest_point[2]                                   # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(np.transpose(dest_point)[:2]))                 # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)                                       # Converts list to a NumPy

                # Transform ball coordinates from frame plane to tactical map plance using the calculated Homography matrix
                # same as aove for players
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point/dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # track ball history
                    if show_b: #check if ball tracking is enabled
                        if len(ball_track_history['src'])>0 :
                            if np.linalg.norm(detected_ball_src_pos-ball_track_history['src'][-1])<ball_track_dist_thresh: #calculates euclidean distance ; checks if less than threshold
                                # Stores the integer coordinates of the ball
                                ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                                ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                            else:
                                # resets the ball track history ie when ball has moves too far , threshold bhanda dherai move bhayidiyo
                                ball_track_history['src']=[(int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1]))]
                                ball_track_history['dst']=[(int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1]))]
                        else:
                            # history nai xaina bhane initialize gardine yesle cahi
                            ball_track_history['src'].append((int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append((int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))
                # ensures the list doesnt grow too long. max_track_length bhanda dherai bhayo bhane oldest lai po gardai janxa
                if len(ball_track_history) > max_track_length:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

                
            ######### Part 2 ########## 
            # Players Team Prediction #
            ###########################

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                                      # Convert frame to RGB
            obj_palette_list = []                                                                   # Initialize players color palette list
            palette_interval = (0,num_pal_colors)                                                   # Color interval to extract from dominant colors palette (1rd to 5th color)

            ## Loop over detected players (label 0) and extract dominant colors palette based on defined interval
            for i, j in enumerate(list(results_players[0].boxes.cls.cpu().numpy())):
                if int(j) == 0:
                    bbox = results_players[0].boxes.xyxy.cpu().numpy()[i,:]                         # Get bbox info (x,y,x,y)
                    obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]       # Crop bbox out of the frame
                    obj_img_w, obj_img_h = obj_img.shape[1], obj_img.shape[0]
                    center_filter_x1 = np.max([(obj_img_w//2)-(obj_img_w//5), 1])
                    center_filter_x2 = (obj_img_w//2)+(obj_img_w//5)
                    center_filter_y1 = np.max([(obj_img_h//3)-(obj_img_h//5), 1])
                    center_filter_y2 = (obj_img_h//3)+(obj_img_h//5)
                    center_filter = obj_img[center_filter_y1:center_filter_y2, 
                                            center_filter_x1:center_filter_x2]
                    obj_pil_img = Image.fromarray(np.uint8(center_filter))                          # Convert to pillow image
                    reduced = obj_pil_img.convert("P", palette=Image.Palette.WEB)                   # Convert to web palette (216 colors)
                    palette = reduced.getpalette()                                                  # Get palette as [r,g,b,r,g,b,...]
                    palette = [palette[3*n:3*n+3] for n in range(256)]                              # Group 3 by 3 = [[r,g,b],[r,g,b],...]
                    color_count = [(n, palette[m]) for n,m in reduced.getcolors()]                  # Create list of palette colors with their frequency
                    RGB_df = pd.DataFrame(color_count, columns = ['cnt', 'RGB']).sort_values(       # Create dataframe based on defined palette interval
                                        by = 'cnt', ascending = False).iloc[
                                            palette_interval[0]:palette_interval[1],:]
                    palette = list(RGB_df.RGB)                                                      # Convert palette to list (for faster processing)
                    
                    # Update detected players color palette list
                    obj_palette_list.append(palette)
            
            ## Calculate distances between each color from every detected player color palette and the predefined teams colors
            players_distance_features = []
            # Loop over detected players extracted color palettes
            for palette in obj_palette_list:
                palette_distance = []
                palette_lab = [skimage.color.rgb2lab([i/255 for i in color]) for color in palette]  # Convert colors to L*a*b* space
                # Loop over colors in palette
                for color in palette_lab:
                    distance_list = []
                    # Loop over predefined list of teams colors
                    for c in color_list_lab:
                        #distance = np.linalg.norm([i/255 - j/255 for i,j in zip(color,c)])
                        distance = skimage.color.deltaE_cie76(color, c)                             # Calculate Euclidean distance in Lab color space
                        distance_list.append(distance)                                              # Update distance list for current color
                    palette_distance.append(distance_list)                                          # Update distance list for current palette
                players_distance_features.append(palette_distance)                                  # Update distance features list

            ## Predict detected players teams based on distance features
            players_teams_list = []
            # Loop over players distance features
            for distance_feats in players_distance_features:
                vote_list=[]
                # Loop over distances for each color 
                for dist_list in distance_feats:
                    #Finds the smallest distance
                    # Retrieves the index of this smallest distance.
                    team_idx = dist_list.index(min(dist_list))//nbr_team_colors                  # Assign team index for current color based on min distance
                    vote_list.append(team_idx)                                                      # Update vote voting list with current color team prediction
                players_teams_list.append(max(vote_list, key=vote_list.count))                      # Predict current player team by vote counting

            # Ball Possession Logic
            if detected_ball_src_pos is not None and len(detected_ppos_src_pts) > 0:
                distances = [np.linalg.norm(detected_ball_src_pos - player_pos) for player_pos in detected_ppos_src_pts]
                min_distance_idx = np.argmin(distances)
                if distances[min_distance_idx] < possession_threshold:
                    possession_team = list(colors_dic.keys())[players_teams_list[min_distance_idx]]
                else:
                    possession_team = None  # Ball is not close enough to any player
            #################### Part 3 #####################
            # Updated Frame & Tactical Map With Annotations #
            #################################################
            
            ball_color_bgr = (0,0,255)                                                                          # Color (GBR) for ball annotation on tactical map
            j=0     
            k=1                                                                                            # Initializing counter of detected players
            palette_box_size = 10                                                                               # Set color box size in pixels (for display)
            annotated_frame = frame                                                                             # Create annotated frame

            # Loop over all detected object by players detection model
            for i in range(bboxes_p.shape[0]):
                conf = confs_p[i]                                                                               # Get confidence of current detected object
                if labels_p[i]==0:   
                    if 'homog' in locals():
                        team_name = list(colors_dic.keys())[players_teams_list[j]]
                        player_pos = (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])) 

                        # Store positions for respective teams
                        if team_name == team_names[0]:
                            team_a_positions.append(player_pos)
                        elif team_name == team_names[1]:
                            team_b_positions.append(player_pos)                                                                          # Display annotation for detected players (label 0)
                    
                    # Display extracted color palette for each detected player
                    if show_pal:
                        palette = obj_palette_list[j]                                                           # Get color palette of the detected player
                        for k, c in enumerate(palette):
                            c_bgr = c[::-1]                                                                     # Convert color to BGR
                            annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,2])+3,             # Add color palette annotation on frame
                                                                    int(bboxes_p[i,1])+k*palette_box_size),
                                                                    (int(bboxes_p[i,2])+palette_box_size,
                                                                    int(bboxes_p[i,1])+(palette_box_size)*(k+1)),
                                                                    c_bgr, -1)
                    team_name = list(colors_dic.keys())[players_teams_list[j]]                                  # Get detected player team prediction
                    color_rgb = colors_dic[team_name][0]                                                        # Get detected player team color
                    color_bgr = color_rgb[::-1]                                                                 # Convert color to bgr
                    if show_p:
                        annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add bbox annotations with team colors
                                                        (int(bboxes_p[i,2]), int(bboxes_p[i,3])), color_bgr, 1)
                        
                        annotated_frame = cv2.putText(annotated_frame, team_name + f" {conf:.2f}",                  # Add team name annotations
                                    (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color_bgr, 2)
                    
                    # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
                    if 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                            radius=5, color=color_bgr, thickness=-1)
                        tac_map_copy = cv2.circle(tac_map_copy, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                            radius=5, color=(0,0,0), thickness=1)
                        if team_name == next(iter(colors_dic)):
                            tac_map_1 = tac_map_copy_team_a
                        elif team_name == list(colors_dic)[1]:
                            tac_map_1 = tac_map_copy_team_b
                        tac_map_1 = cv2.circle(tac_map_1, (int(pred_dst_pts[j][0]),int(pred_dst_pts[j][1])),
                                            radius=2, color=color_bgr, thickness=1)
                    if team_name == next(iter(colors_dic)):
                        tac_map_copy_team_a = tac_map_1
                    elif team_name == list(colors_dic)[1]:
                        tac_map_copy_team_b = tac_map_1
                    tac_map_1 = tac_map.copy()
                    j+=1  
                    k=j+1                                                                                     # Update players counter
                else:                                                                                           # Display annotation for otehr detections (label 1, 2)
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_p[i,0]), int(bboxes_p[i,1])),  # Add white colored bbox annotations
                                                    (int(bboxes_p[i,2]), int(bboxes_p[i,3])), (255,255,255), 1)
                    annotated_frame = cv2.putText(annotated_frame, labels_dic[labels_p[i]] + f" {conf:.2f}",    # Add white colored label text annotations
                                (int(bboxes_p[i,0]), int(bboxes_p[i,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,255,255), 2)

                    # Add tactical map ball postion annotation if detected
                    if detected_ball_src_pos is not None and 'homog' in locals():
                        tac_map_copy = cv2.circle(tac_map_copy, (int(detected_ball_dst_pos[0]), 
                                                    int(detected_ball_dst_pos[1])), radius=5, 
                                                    color=ball_color_bgr, thickness=3)
            if show_k:
                for i in range(bboxes_k.shape[0]):
                    annotated_frame = cv2.rectangle(annotated_frame, (int(bboxes_k[i,0]), int(bboxes_k[i,1])),  # Add bbox annotations with team colors
                                                (int(bboxes_k[i,2]), int(bboxes_k[i,3])), (0,0,0), 1)
            # Plot the tracks
            if len(ball_track_history['src'])>0:
                points = np.hstack(ball_track_history['dst']).astype(np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points], isClosed=False, color=(0, 0, 100), thickness=2)

            # Add Ball Possession Annotation
            if possession_team is not None:
                possession_color = colors_dic[possession_team][0][::-1]  # Convert RGB to BGR
                cv2.putText(annotated_frame, f"Possession: {possession_team}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, possession_color, 2)
            else:
                cv2.putText(annotated_frame, "Possession: None", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Combine annotated frame and tactical map in one image with colored border separation
            border_color = [255,255,255]                                                                        # Set border color (BGR)
            annotated_frame=cv2.copyMakeBorder(annotated_frame, 40, 10, 10, 10,                                 # Add borders to annotated frame
                                                cv2.BORDER_CONSTANT, value=border_color)
            tac_map_copy = cv2.copyMakeBorder(tac_map_copy, 70, 50, 10, 10, cv2.BORDER_CONSTANT,                # Add borders to tactical map 
                                            value=border_color)      
            tac_map_copy = cv2.resize(tac_map_copy, (tac_map_copy.shape[1], annotated_frame.shape[0]))          # Resize tactical map
            final_img = cv2.hconcat((annotated_frame, tac_map_copy))                                            # Concatenate both images
            ## Add info annotation
            cv2.putText(final_img, "Tactical Map", (1370,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)

            new_frame_time = time.time()                                                                        # Get time after finished processing current frame
            fps = 1/(new_frame_time-prev_frame_time)                                                            # Calculate FPS as 1/(frame proceesing duration)
            prev_frame_time = new_frame_time                                                                    # Save current time to be used in next frame
            cv2.putText(final_img, "FPS: " + str(int(fps)), (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
            
            # Display the annotated frame
            stframe.image(final_img, channels="BGR")
            #cv2.imshow("YOLOv8 Inference", frame)
            if save_output:
                output.write(cv2.resize(final_img, (width, height)))

    # After the loop, generate and save heatmaps
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
        return heatmap_img 

    cv2.imwrite(os.path.join("./outputs","output_image_team_A.jpg"),tac_map_copy_team_a)
    cv2.imwrite(os.path.join("./outputs","output_image_team_B.jpg"),tac_map_copy_team_b)
    
   # Generate and save heatmaps for both teams
    heatmap_team_a = create_heatmap(team_a_positions, tac_map, './outputs/heatmap_team_A.jpg')
    heatmap_team_b = create_heatmap(team_b_positions, tac_map, './outputs/heatmap_team_B.jpg')
    
    # Update Streamlit display
    heatmap1.image(heatmap_team_a, channels="BGR")
    heatmap2.image(heatmap_team_b, channels="BGR")
    trace1.image(tac_map_copy_team_a, channels="BGR")
    trace2.image(tac_map_copy_team_b, channels="BGR")
    # Remove progress bar and return        
    st_prog_bar.empty()
    return True


        
    


    