import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import altair as alt
import io

# Page configuration and style
st.set_page_config(
    page_title="Glacier Growth Detection",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    .stApp {
        max-width: 1600px;
        margin: 0 auto;
    }
    .title-container {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2em;
        opacity: 0.8;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .green-highlight {
        color: #10B981;
        font-weight: bold;
    }
    .red-highlight {
        color: #EF4444;
        font-weight: bold;
    }
    .image-container {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9em;
        color: #6B7280;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .legend-color {
        width: 15px;
        height: 15px;
        margin-right: 8px;
        border-radius: 3px;
    }
    .lake-stats {
        background-color: #EBF4FF;
        border-radius: 8px;
        padding: 15px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Custom title with gradient background
st.markdown("""
<div class="title-container">
    <h1>Monitoring and Assessment of GLOF ❄️</h1>
</div>
""", unsafe_allow_html=True)

# Load YOLO segmentation model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

def get_detection_overlay(image):
    try:
        results = model.predict(source=image, save=False)
        overlay = image.copy()
        lake_count = 0  # Initialize lake counter
        
        for r in results:
            lake_count = len(r)  # Count number of detected instances
            for c in r:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
        
        return overlay, lake_count
    except Exception as e:
        st.error(f"Error during model inference: {e}")
        return None, 0

def get_binary_mask(image):
    try:
        results = model.predict(source=image, save=False)
        binary_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        lake_count = 0  # Initialize lake counter
        lake_areas = []  # Track individual lake areas
        
        for r in results:
            lake_count = len(r)  # Count number of detected instances
            for c in r:
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(binary_mask, [contour], -1, 255, thickness=cv2.FILLED)
                # Calculate area of this lake
                area = cv2.contourArea(contour)
                lake_areas.append(area)
        
        return binary_mask, lake_count, lake_areas
    except Exception as e:
        st.error(f"Error generating binary mask: {e}")
        return None, 0, []

def calculate_growth_and_overlay(image1, image2):
    mask1, lake_count1, lake_areas1 = get_binary_mask(image1)
    mask2, lake_count2, lake_areas2 = get_binary_mask(image2)

    if mask1 is None or mask2 is None:
        return None, None, None, None, None, None, None, None, 0, 0, [], []

    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]), interpolation=cv2.INTER_NEAREST)

    intersection_mask = cv2.bitwise_and(mask1, mask2)
    growth_mask = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))

    intersection_color = (0, 255, 0)  # Green for consistent area
    growth_color = (0, 0, 255)        # Red for growth area

    image2_with_intersection = overlay_mask_on_image(image2, intersection_mask, intersection_color, alpha=0.5)
    image2_with_growth = overlay_mask_on_image(image2_with_intersection, growth_mask, growth_color, alpha=0.5)

    total_area_mask1 = np.sum(mask1 == 255)
    total_area_mask2 = np.sum(mask2 == 255)
    intersection_area = np.sum(intersection_mask == 255)
    growth_area = np.sum(growth_mask == 255)

    if total_area_mask1 > 0:
        growth_percentage = (growth_area / total_area_mask1) * 100
    else:
        growth_percentage = 0

    # Growth status and risk level
    growth_status = "Increasing" if total_area_mask2 > total_area_mask1 else "Decreasing"
    risk_level = "Risky" if growth_percentage > st.session_state.get("risk_threshold", 10) else "Not Risky"

    return image2_with_growth, total_area_mask1, total_area_mask2, intersection_area, growth_area, growth_percentage, growth_status, risk_level, lake_count1, lake_count2, lake_areas1, lake_areas2

def overlay_mask_on_image(image, mask, color, alpha=0.5):
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = color
    return cv2.addWeighted(colored_mask, alpha, image, 1 - alpha, 0)

# Initialize session state variables
if 'baseline_processed' not in st.session_state:
    st.session_state.baseline_processed = False
if 'current_processed' not in st.session_state:
    st.session_state.current_processed = False
if 'baseline_overlay' not in st.session_state:
    st.session_state.baseline_overlay = None
if 'baseline_lake_count' not in st.session_state:
    st.session_state.baseline_lake_count = 0
if 'current_overlay' not in st.session_state:
    st.session_state.current_overlay = None
if 'current_lake_count' not in st.session_state:
    st.session_state.current_lake_count = 0
if 'image1_data' not in st.session_state:
    st.session_state.image1_data = None
if 'image2_data' not in st.session_state:
    st.session_state.image2_data = None

# Upload section
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Baseline Image")
    uploaded_image1 = st.file_uploader("Choose the first image", type=["jpg", "jpeg", "png"], key="img1")
    
    if uploaded_image1:
        # Store the image data in session state
        file_bytes = uploaded_image1.getvalue()
        
        # Reset processed state if new image is uploaded
        if 'last_baseline_image' not in st.session_state or st.session_state.last_baseline_image != uploaded_image1.name:
            st.session_state.baseline_processed = False
            st.session_state.last_baseline_image = uploaded_image1.name
            st.session_state.image1_data = file_bytes
        
        # Read image for display
        image1 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        st.image(image1_rgb, caption="Baseline Image", use_container_width=True)
        
        # Automatically run detection when image is uploaded
        if not st.session_state.baseline_processed:
            with st.spinner("Detecting lakes in baseline image..."):
                baseline_overlay, baseline_lake_count = get_detection_overlay(image1)
                if baseline_overlay is not None:
                    st.session_state.baseline_overlay = baseline_overlay
                    st.session_state.baseline_lake_count = baseline_lake_count
                    st.session_state.baseline_processed = True
                    
        # Display detection results if available
        if st.session_state.baseline_processed and st.session_state.baseline_overlay is not None:
            baseline_overlay_rgb = cv2.cvtColor(st.session_state.baseline_overlay, cv2.COLOR_BGR2RGB)
            st.image(baseline_overlay_rgb, caption=f"Detected Lakes: {st.session_state.baseline_lake_count}", use_container_width=True)

with col2:
    st.markdown("### Current Image")
    uploaded_image2 = st.file_uploader("Choose the second image", type=["jpg", "jpeg", "png"], key="img2")
    
    if uploaded_image2:
        # Store the image data in session state
        file_bytes = uploaded_image2.getvalue()
        
        # Reset processed state if new image is uploaded
        if 'last_current_image' not in st.session_state or st.session_state.last_current_image != uploaded_image2.name:
            st.session_state.current_processed = False
            st.session_state.last_current_image = uploaded_image2.name
            st.session_state.image2_data = file_bytes
            
        # Read image for display
        image2 = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        st.image(image2_rgb, caption="Current Image", use_container_width=True)
        
        # Automatically run detection when image is uploaded
        if not st.session_state.current_processed:
            with st.spinner("Detecting lakes in current image..."):
                current_overlay, current_lake_count = get_detection_overlay(image2)
                if current_overlay is not None:
                    st.session_state.current_overlay = current_overlay
                    st.session_state.current_lake_count = current_lake_count
                    st.session_state.current_processed = True
                    
        # Display detection results if available
        if st.session_state.current_processed and st.session_state.current_overlay is not None:
            current_overlay_rgb = cv2.cvtColor(st.session_state.current_overlay, cv2.COLOR_BGR2RGB)
            st.image(current_overlay_rgb, caption=f"Detected Lakes: {st.session_state.current_lake_count}", use_container_width=True)

# Risk threshold slider
st.slider("Set growth percentage threshold for risk level", 0, 100, 10, key="risk_threshold")

# Analysis section - Now we'll provide a button for the full analysis
if uploaded_image1 and uploaded_image2 and st.session_state.image1_data is not None and st.session_state.image2_data is not None:
    if st.button("🔍 Run Full Analysis"):
        with st.spinner("🧠 AI is analyzing glacier and lake changes..."):
            # Use stored image data from session state to decode fresh images
            image1 = cv2.imdecode(np.frombuffer(st.session_state.image1_data, np.uint8), cv2.IMREAD_COLOR)
            image2 = cv2.imdecode(np.frombuffer(st.session_state.image2_data, np.uint8), cv2.IMREAD_COLOR)
            
            result_image, area1, area2, intersection, growth, growth_percent, growth_status, risk_level, lake_count1, lake_count2, lake_areas1, lake_areas2 = calculate_growth_and_overlay(image1, image2)

            if result_image is not None:
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

                # Add warning message for multiple lakes
                if lake_count1 > 1 or lake_count2 > 1:
                    st.warning("⚠️ **Multiple lakes detected!** This analysis shows combined results for all lakes and not individual lakes. Results may be less accurate for tracking specific lake changes.")
                
                # Display results
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📊 Analysis Results")

                # Lake detection summary
                st.markdown("### 🌊 Lake Detection Summary")
                lake_col1, lake_col2, lake_col3 = st.columns(3)
                
                with lake_col1:
                    st.metric("Lakes in Baseline", lake_count1)
                
                with lake_col2:
                    st.metric("Lakes in Current", lake_count2)
                
                with lake_col3:
                    lake_change = lake_count2 - lake_count1
                    st.metric("Lake Change", lake_change, delta=lake_change)

                # Lake area statistics
                if lake_areas1 and lake_areas2:
                    avg_area1 = sum(lake_areas1) / len(lake_areas1) if lake_areas1 else 0
                    avg_area2 = sum(lake_areas2) / len(lake_areas2) if lake_areas2 else 0
                    max_area1 = max(lake_areas1) if lake_areas1 else 0
                    max_area2 = max(lake_areas2) if lake_areas2 else 0
                    
                    st.markdown('<div class="lake-stats">', unsafe_allow_html=True)
                    lake_stats_col1, lake_stats_col2 = st.columns(2)
                    
                    with lake_stats_col1:
                        st.metric("Avg Lake Size (Baseline)", f"{avg_area1:.1f} px²")
                        st.metric("Max Lake Size (Baseline)", f"{max_area1:.1f} px²")
                    
                    with lake_stats_col2:
                        st.metric("Avg Lake Size (Current)", f"{avg_area2:.1f} px²")
                        st.metric("Max Lake Size (Current)", f"{max_area2:.1f} px²")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

                col5, col6 = st.columns([3, 2])
                with col5:
                    st.markdown("### Growth and Persistent Glacier Areas")
                    st.image(result_image_rgb, caption="Glacier Change Visualization", use_container_width=True)

                    # Legend
                    st.markdown("#### Result:")
                    st.markdown("""
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(0, 255, 0, 0.5);"></div>
                        <div>Persistent Glacier Area</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: rgba(255, 0, 0, 0.5);"></div>
                        <div>New Growth Area</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col6:
                    st.markdown("### Key Metrics")
                    st.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <div class="metric-label">Growth Status</div>
                        <div class="metric-value {'green-highlight' if growth_status == 'Increasing' else 'red-highlight'}">{growth_status}</div>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <div class="metric-label">Risk Assessment</div>
                        <div class="metric-value {'red-highlight' if risk_level == 'Risky' else 'green-highlight'}">{risk_level}</div>
                    </div>
                    <div style="margin-bottom: 20px;">
                        <div class="metric-label">Growth Percentage</div>
                        <div class="metric-value">{growth_percent:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Area Measurements Section
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("## 📈 Area Measurements")

                # Create data for bar chart
                data = pd.DataFrame({
                    'Metric': ['Baseline Area', 'Current Area', 'Persistent Area', 'New Growth'],
                    'Area': [area1, area2, intersection, growth],
                    'Color': ['#6366F1', '#8B5CF6', '#22C55E', '#EF4444']
                })

                # Create bar chart
                chart = alt.Chart(data).mark_bar().encode(
                    x=alt.X('Metric', sort=None),
                    y='Area',
                    color=alt.Color('Color:N', scale=None)
                ).properties(
                    height=300
                )

                col7, col8 = st.columns([3, 2])
                with col7:
                    st.altair_chart(chart, use_container_width=True)

                with col8:
                    # Area statistics in metrics
                    metric_col1, metric_col2 = st.columns(2)
                    with metric_col1:
                        st.metric("Baseline Area", f"{area1:,} px²")
                        st.metric("Persistent Area", f"{intersection:,} px²")
                    with metric_col2:
                        st.metric("Current Area", f"{area2:,} px²")
                        st.metric("New Growth", f"{growth:,} px²")

                st.markdown('</div>', unsafe_allow_html=True)

                # Lake Distribution Section
                if lake_areas1 and lake_areas2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("## 🌊 Lake Size Distribution")
                    
                    # Create histogram data for lake areas
                    lake_data1 = pd.DataFrame({
                        'Area': lake_areas1,
                        'Image': ['Baseline'] * len(lake_areas1)
                    })
                    lake_data2 = pd.DataFrame({
                        'Area': lake_areas2,
                        'Image': ['Current'] * len(lake_areas2)
                    })
                    lake_data = pd.concat([lake_data1, lake_data2])
                    
                    # Create histogram
                    bins = alt.Bin(maxbins=10)
                    histogram = alt.Chart(lake_data).mark_bar(opacity=0.7).encode(
                        x=alt.X('Area:Q', bin=bins, title='Lake Size (px²)'),
                        y=alt.Y('count()', stack=None),
                        color='Image:N'
                    ).properties(
                        height=300
                    )
                    
                    st.altair_chart(histogram, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)