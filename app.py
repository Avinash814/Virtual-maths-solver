import streamlit as st
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
import google.generativeai as genai

# AI model configuration
genai.configure(api_key="AIzaSyDkwS9HNNZ2xg1ifRrq2p0AD312uCa0Zgo")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

# Streamlit app setup
st.set_page_config(
    page_title="Virtual Math Solver",
    page_icon="📊"
)

st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #333;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    <h1 class="title">Virtual Math Solver</h1>
""", unsafe_allow_html=True)

# Session state initialization
if 'prev_pos' not in st.session_state:
    st.session_state.prev_pos = None
if 'canvas' not in st.session_state:
    st.session_state.canvas = None
if 'ai_response' not in st.session_state:
    st.session_state.ai_response = ""
if 'run' not in st.session_state:
    st.session_state.run = True
if 'lines' not in st.session_state:
    st.session_state.lines = []  # Store drawn lines for erasing

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    left_hand, right_hand = None, None
    for hand in hands:
        if hand["type"] == "Right":
            right_hand = hand
        elif hand["type"] == "Left":
            left_hand = hand
    return left_hand, right_hand

# Function to draw on the canvas
def draw(info, prev_pos, canvas, lines):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Left index finger raised
        current_pos = lmList[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, tuple(prev_pos), tuple(current_pos), (255, 0, 255), 10)
        lines.append((tuple(prev_pos), tuple(current_pos)))  # Store the line coordinates
        prev_pos = current_pos
    return current_pos, canvas, lines

# Function to erase the last line drawn
def eraseLastLine(canvas, lines):
    if lines:
        last_line = lines.pop()  # Remove the last line from the list
        cv2.line(canvas, last_line[0], last_line[1], (0, 0, 0), 10)  # Draw black line over it
    return canvas, lines

# Function to clear the entire canvas
def clearCanvas():
    return np.zeros_like(st.session_state.canvas), []

# Function to send the canvas to the AI model
def sendToAI(model, canvas):
    resized_canvas = cv2.resize(canvas, (256, 256))
    pil_image = Image.fromarray(resized_canvas)
    try:
        response = model.generate_content([f"Solve this problem: ", pil_image])
        return response.text
    except Exception as e:
        st.error(f"Error sending to AI: {e}")
        return ""

# Sidebar buttons and instructions
clear_canvas = st.sidebar.button("Clear Canvas")
quit_button = st.sidebar.button("Quit")

st.sidebar.markdown("""
    <h2 style='color: #333;'>Controls</h2>
    <ul>
        <li><strong>Draw:</strong> Use your left hand's index finger to draw.</li>
        <li><strong>Erase last line:</strong> Raise four fingers of your left hand.</li>
        <li><strong>Submit:</strong> Raise four fingers of your right hand to submit the question.</li>
        <li><strong>Clear:</strong> Raise your right hand's thumb to clear the canvas.</li>
    </ul>
""", unsafe_allow_html=True)


# Add your name at the bottom of the sidebar
st.sidebar.caption("Created by :red[Avinash] ✨")

canvas_placeholder = st.empty()
response_placeholder = st.empty()

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    st.error("Error: Could not open video capture")
else:
    while st.session_state.run:
        success, img = cap.read()
        if not success or img is None:
            st.error("Error: Could not read frame from camera")
            break

        img = cv2.flip(img, 1)
        if st.session_state.canvas is None:
            st.session_state.canvas = np.zeros_like(img)

        # Clear canvas if requested
        if clear_canvas:
            st.session_state.canvas, st.session_state.lines = clearCanvas()

        # Get hand info
        left_hand, right_hand = getHandInfo(img)

        # Draw with left hand
        if left_hand:
            left_fingers, left_lmList = detector.fingersUp(left_hand), left_hand["lmList"]
            st.session_state.prev_pos, st.session_state.canvas, st.session_state.lines = draw((left_fingers, left_lmList), st.session_state.prev_pos, st.session_state.canvas, st.session_state.lines)

        # Erase the last line with four fingers raised on the left hand
        if left_hand:
            left_fingers, left_lmList = detector.fingersUp(left_hand), left_hand["lmList"]
            if left_fingers == [0, 1, 1, 1, 1]:  # Four fingers raised on the left hand
                st.session_state.canvas, st.session_state.lines = eraseLastLine(st.session_state.canvas, st.session_state.lines)

        # Submit the drawing to the AI model with right hand
        if right_hand:
            right_fingers, right_lmList = detector.fingersUp(right_hand), right_hand["lmList"]
            if right_fingers == [0, 1, 1, 1, 1]:  # Four fingers raised on the right hand
                st.session_state.ai_response = sendToAI(model, st.session_state.canvas)
            elif right_fingers == [1, 0, 0, 0, 0]:  # Thumb raised on the right hand
                st.session_state.canvas, st.session_state.lines = clearCanvas()

        image_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
        image_combined = cv2.cvtColor(image_combined, cv2.COLOR_BGR2RGB)

        # Display the canvas
        with canvas_placeholder.container():
            st.image(image_combined, caption="Virtual Math Solver", use_column_width=True)

        # Display the AI response
        if st.session_state.ai_response:
            with response_placeholder.container():
                st.markdown(f"""
                    <div style='padding: 10px; background-color: #e9ecef; border-radius: 5px;'>
                        <h3 style='color: #007bff;'>AI Response</h3>
                        <p>{st.session_state.ai_response}</p>
                    </div>
                """, unsafe_allow_html=True)

        

        # Stop the loop if the quit button is pressed
        if quit_button:
            st.session_state.run = False

    cap.release()
    cv2.destroyAllWindows()

    
