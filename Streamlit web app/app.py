import tempfile
import numpy as np
import glob
import os
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import cv2
from ultralytics import YOLO
from detection import create_colors_info, detect
import matplotlib.pyplot as plt  # Added for heatmap
from scipy.ndimage import gaussian_filter  # Added for heatmap

def main():
    st.set_page_config(page_title="Major Project", layout="wide", initial_sidebar_state="collapsed")
    st.title("Football Players Detection With Team Prediction & Tactical Map")

    # Initialize session state for team colors and commentary (from second app.py)
    demo_team_info = {
        "Demo 1": {
            "team1_name": "France",
            "team2_name": "Switzerland",
            "team1_p_color": '#1E2530',
            "team1_gk_color": '#F5FD15',
            "team2_p_color": '#FBFCFA',
            "team2_gk_color": '#B1FCC4',
        },
        "Demo 2": {
            "team1_name": "Chelsea",
            "team2_name": "Manchester City",
            "team1_p_color": '#29478A',
            "team1_gk_color": '#DC6258',
            "team2_p_color": '#90C8FF',
            "team2_gk_color": '#BCC703',
        }
    }
    st.sidebar.title("Upload")
    demo_selected = st.sidebar.radio(label="Select Demo Video", options=["Demo 1", "Demo 2"], horizontal=True)
    selected_team_info = demo_team_info[demo_selected]
    team1_name = selected_team_info["team1_name"]
    team2_name = selected_team_info["team2_name"]

    # Initialize session state for colors and commentary (from second app.py)
    if f"{team1_name} P color" not in st.session_state:
        st.session_state[f"{team1_name} P color"] = selected_team_info["team1_p_color"]
    if f"{team1_name} GK color" not in st.session_state:
        st.session_state[f"{team1_name} GK color"] = selected_team_info["team1_gk_color"]
    if f"{team2_name} P color" not in st.session_state:
        st.session_state[f"{team2_name} P color"] = selected_team_info["team2_p_color"]
    if f"{team2_name} GK color" not in st.session_state:
        st.session_state[f"{team2_name} GK color"] = selected_team_info["team2_gk_color"]
    if 'commentary' not in st.session_state:
        st.session_state['commentary'] = []
    if 'heatmap_analysis' not in st.session_state:
        st.session_state['heatmap_analysis'] = []

    ## Sidebar Setup
    st.sidebar.subheader("Video Upload")
    input_vide_file = st.sidebar.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi', 'm4v', 'asf'])

    demo_vid_paths = {
        "Demo 1": './demo_vid_11.mp4',
        "Demo 2": './demo_vid_2.mp4'
    }
    demo_vid_path = demo_vid_paths[demo_selected]

    tempf = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    if not input_vide_file:
        tempf.name = demo_vid_path
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Demo video')
        st.sidebar.video(demo_bytes)
    else:
        tempf.write(input_vide_file.read())
        demo_vid = open(tempf.name, 'rb')
        demo_bytes = demo_vid.read()

        st.sidebar.text('Input video')
        st.sidebar.video(demo_bytes)

    # Load the YOLOv8 players detection model
    model_players = YOLO("../models/Yolo8L Players/weights/best.pt")
    # Load the YOLOv8 field keypoints detection model
    model_keypoints = YOLO("../models/Yolo8M Field Keypoints/weights/best.pt")

    ## Page Setup
    # Add tab5 for Commentary (from second app.py)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["About Project", "Team Colors", "Model Hyperparameters & Detection", "Analysis", "Commentary"])

    with tab1:
        with open("../readme.md", 'r') as f:
            readme_line = f.readlines()
            readme_buffer = []
            resource_files = [os.path.basename(x) for x in glob.glob(f'Resources/*')]
        for line in readme_line:
            readme_buffer.append(line)
            for image in resource_files:
                if image in line:
                    st.markdown(''.join(readme_buffer[:-1]))
                    st.image(f'Resources/{image}')
                    readme_buffer.clear()
        st.markdown(''.join(readme_buffer))

    with tab2:
        t1col1, t1col2 = st.columns([1, 1])
        with t1col1:
            cap_temp = cv2.VideoCapture(tempf.name)
            frame_count = int(cap_temp.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_nbr = st.slider(label="Select frame", min_value=1, max_value=frame_count, step=1, help="Select frame to pick team colors from")
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, frame_nbr)
            success, frame = cap_temp.read()
            with st.spinner('Detecting players in selected frame..'):
                results = model_players(frame, conf=0.7)
                bboxes = results[0].boxes.xyxy.cpu().numpy()
                labels = results[0].boxes.cls.cpu().numpy()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detections_imgs_list = []
                detections_imgs_grid = []
                padding_img = np.ones((80, 60, 3), dtype=np.uint8) * 255
                for i, j in enumerate(list(labels)):
                    if int(j) == 0:
                        bbox = bboxes[i, :]
                        obj_img = frame_rgb[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                        obj_img = cv2.resize(obj_img, (60, 80))
                        detections_imgs_list.append(obj_img)
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2)])
                detections_imgs_grid.append([detections_imgs_list[i] for i in range(len(detections_imgs_list)//2, len(detections_imgs_list))])
                if len(detections_imgs_list) % 2 != 0:
                    detections_imgs_grid[0].append(padding_img)
                concat_det_imgs_row1 = cv2.hconcat(detections_imgs_grid[0])
                concat_det_imgs_row2 = cv2.hconcat(detections_imgs_grid[1])
                concat_det_imgs = cv2.vconcat([concat_det_imgs_row1, concat_det_imgs_row2])
            st.write("Team Names")
            team1_name = st.text_input(label='First Team Name', value=selected_team_info["team1_name"])
            team2_name = st.text_input(label='Second Team Name', value=selected_team_info["team2_name"])
            st.write("Detected players")
            value = streamlit_image_coordinates(concat_det_imgs, key="numpy")
            st.markdown('---')
            radio_options = [f"{team1_name} P color", f"{team1_name} GK color", f"{team2_name} P color", f"{team2_name} GK color"]
            active_color = st.radio(label="Select which team color to pick from the image above", options=radio_options, horizontal=True,
                                    help="Chose team color you want to pick and click on the image above to pick the color. Colors will be displayed in boxes below.")
            if value is not None:
                picked_color = concat_det_imgs[value['y'], value['x'], :]
                st.session_state[f"{active_color}"] = '#%02x%02x%02x' % tuple(picked_color)
            st.write("Boxes below can be used to manually adjust selected colors.")
            cp1, cp2, cp3, cp4 = st.columns([1, 1, 1, 1])
            with cp1:
                hex_color_1 = st.session_state[f"{team1_name} P color"] if f"{team1_name} P color" in st.session_state else selected_team_info["team1_p_color"]
                team1_p_color = st.color_picker(label=' ', value=hex_color_1, key='t1p')
                st.session_state[f"{team1_name} P color"] = team1_p_color
            with cp2:
                hex_color_2 = st.session_state[f"{team1_name} GK color"] if f"{team1_name} GK color" in st.session_state else selected_team_info["team1_gk_color"]
                team1_gk_color = st.color_picker(label=' ', value=hex_color_2, key='t1gk')
                st.session_state[f"{team1_name} GK color"] = team1_gk_color
            with cp3:
                hex_color_3 = st.session_state[f"{team2_name} P color"] if f"{team2_name} P color" in st.session_state else selected_team_info["team2_p_color"]
                team2_p_color = st.color_picker(label=' ', value=hex_color_3, key='t2p')
                st.session_state[f"{team2_name} P color"] = team2_p_color
            with cp4:
                hex_color_4 = st.session_state[f"{team2_name} GK color"] if f"{team2_name} GK color" in st.session_state else selected_team_info["team2_gk_color"]
                team2_gk_color = st.color_picker(label=' ', value=hex_color_4, key='t2gk')
                st.session_state[f"{team2_name} GK color"] = team2_gk_color
        st.markdown('---')

        with t1col2:
            extracted_frame = st.empty()
            extracted_frame.image(frame, use_column_width=True, channels="BGR")

    colors_dic, color_list_lab = create_colors_info(team1_name, st.session_state[f"{team1_name} P color"], st.session_state[f"{team1_name} GK color"],
                                                     team2_name, st.session_state[f"{team2_name} P color"], st.session_state[f"{team2_name} GK color"])

    with tab4:
        st.title("Heatmap")
        st.markdown("---")
        ccol1, ccol2 = st.columns([1, 1])
        with ccol1:
            st.write(f'{team1_name} HeatMap')
            heatmap1 = st.empty()
            # Add dynamic image display (from second app.py)
            if 'heatmap1_image' in st.session_state:
                heatmap1.image(st.session_state['heatmap1_image'], use_column_width=True)
        with ccol2:
            st.write(f'{team2_name} HeatMap')
            heatmap2 = st.empty()
            # Add dynamic image display (from second app.py)
            if 'heatmap2_image' in st.session_state:
                heatmap2.image(st.session_state['heatmap2_image'], use_column_width=True)
        st.title("Player Tracing")
        
        dcol1, dcol2 = st.columns([1, 1])
        with dcol1:
            st.write(f'{team1_name} Trace')
            trace1 = st.empty()
            # Add dynamic image display (from second app.py)
            if 'trace1_image' in st.session_state:
                trace1.image(st.session_state['trace1_image'], use_column_width=True)
        with dcol2:
            st.write(f'{team2_name} Trace')
            trace2 = st.empty()
            # Add dynamic image display (from second app.py)
            if 'trace2_image' in st.session_state:
                trace2.image(st.session_state['trace2_image'], use_column_width=True)
        st.markdown("---")
        # Add heatmap analysis section (from second app.py)
        st.write("### Heatmap Analysis")
        if 'heatmap_analysis' in st.session_state and st.session_state['heatmap_analysis']:
            for comment in st.session_state['heatmap_analysis']:
                st.write(comment)
        else:
            st.write("Run detection to generate heatmap analysis.")

    with tab3:
        t2col1, t2col2 = st.columns([1, 1])
        with t2col1:
            player_model_conf_thresh = st.slider('PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.3)
            keypoints_model_conf_thresh = st.slider('Field Keypoints PLayers Detection Confidence Threshold', min_value=0.0, max_value=1.0, value=0.35)
            keypoints_displacement_mean_tol = st.slider('Keypoints Displacement RMSE Tolerance (pixels)', min_value=-1, max_value=100, value=7,
                                                         help="Indicates the maximum allowed average distance between the position of the field keypoints\
                                                           in current and previous detections. It is used to determine wether to update homography matrix or not. ")
            detection_hyper_params = {
                0: player_model_conf_thresh,
                1: keypoints_model_conf_thresh,
                2: keypoints_displacement_mean_tol
            }
        with t2col2:
            num_pal_colors = st.slider(label="Number of palette colors", min_value=1, max_value=5, step=1, value=3,
                                    help="How many colors to extract form detected players bounding-boxes? It is used for team prediction.")
            st.markdown("---")
            save_output = st.checkbox(label='Save output', value=False)
            if save_output:
                output_file_name = st.text_input(label='File Name (Optional)', placeholder='Enter output video file name.')
            else:
                output_file_name = None
        st.markdown("---")

        bcol1, bcol2 = st.columns([1, 1])
        with bcol1:
            nbr_frames_no_ball_thresh = 30
            ball_track_dist_thresh = 100
            max_track_length = 35
            ball_track_hyperparams = {
                0: nbr_frames_no_ball_thresh,
                1: ball_track_dist_thresh,
                2: max_track_length
            }
        with bcol2:
            st.write("Annotation options:")
            bcol21t, bcol22t = st.columns([1, 1])
            with bcol21t:
                show_k = st.toggle(label="Show Keypoints Detections", value=False)
                show_p = st.toggle(label="Show Players Detections", value=True)
            with bcol22t:
                show_pal = st.toggle(label="Show Color Palettes", value=True)
                show_b = st.toggle(label="Show Ball Tracks", value=True)
                show_possession = st.toggle(label="Show Ball Possession", value=True)  # New toggle for possession
            plot_hyperparams = {
                0: show_k,
                1: show_pal,
                2: show_b,
                3: show_p,
                4: show_possession  # Add possession to plot_hyperparams
            }
            st.markdown('---')
            bcol21, bcol22, bcol23, bcol24 = st.columns([1.5, 1, 1, 1])
            with bcol21:
                st.write('')
            with bcol22:
                ready = True if (team1_name == '') or (team2_name == '') else False
                start_detection = st.button(label='Start Detection', disabled=ready)
            with bcol23:
                stop_btn_state = True if not start_detection else False
                stop_detection = st.button(label='Stop Detection', disabled=stop_btn_state)
            with bcol24:
                st.write('')

        stframe = st.empty()
        cap = cv2.VideoCapture(tempf.name)
        status = False

    # Create a placeholder for real-time commentary in the Commentary tab (from second app.py)
    with tab5:
        st.title("Match Commentary")
        commentary_placeholder = st.empty()  # Placeholder for real-time updates
        if 'commentary' in st.session_state and st.session_state['commentary']:
            with commentary_placeholder.container():
                st.write("### Analyst Commentary Log")
                for comment in st.session_state['commentary']:
                    if not comment.startswith("Post-Match Analysis: Heatmap Insights"):
                        st.write(comment)
        else:
            commentary_placeholder.write("Run detection to generate analyst commentary.")

    # Update detection logic to include commentary and dynamic image updates (from second app.py)
    if start_detection and not stop_detection:
        st.session_state['running'] = True  # Add running state
        st.toast(f'Detection Started!')
        try:
            # Pass commentary_placeholder to detect function
            status = detect(cap, stframe, heatmap1, heatmap2, trace1, trace2, commentary_placeholder,
                            output_file_name, save_output, model_players, model_keypoints,
                            detection_hyper_params, ball_track_hyperparams, plot_hyperparams,
                            num_pal_colors, colors_dic, color_list_lab)
            # Update session state with images and analysis after detection
            if status:
                if 'heatmap1_image' in st.session_state and 'heatmap2_image' in st.session_state:
                    heatmap1.image(st.session_state['heatmap1_image'], use_column_width=True)
                    heatmap2.image(st.session_state['heatmap2_image'], use_column_width=True)
                if 'trace1_image' in st.session_state and 'trace2_image' in st.session_state:
                    trace1.image(st.session_state['trace1_image'], use_column_width=True)
                    trace2.image(st.session_state['trace2_image'], use_column_width=True)
        except Exception as e:
            st.error(f"Detection failed: {str(e)}")
            status = False
    elif stop_detection:
        st.session_state['running'] = False  # Add running state
        st.toast(f'Detection Stopped!')
        try:
            cap.release()
        except:
            pass
    else:
        try:
            cap.release()
        except:
            pass

    if status:
        st.toast(f'Detection Completed!')
        cap.release()
        st.experimental_rerun()  # Add rerun to refresh UI (from second app.py)

if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")