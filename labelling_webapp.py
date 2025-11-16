import streamlit as st
import cv2
from pathlib import Path
import shutil
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Face Shape Labeling Tool",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        height: 55px;
        font-size: 18px;
        font-weight: bold;
        margin: 5px 0;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class StreamlitLabellingApp:
    """Streamlit app for face shape labeling"""
    
    def __init__(self):
        self.input_folder = Path(r'C:\Users\aniru\opencv_project\dataset\train')
        self.output_folder = Path(r'C:\Users\aniru\opencv_project\dataset\manually_labeled_dataset')
        self.shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        
        # Create output directories
        for shape in self.shapes:
            (self.output_folder / shape).mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self):
        """Get all unlabeled image files"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        image_files = []
        
        # Get images from all subfolders
        for ext in image_extensions:
            image_files.extend(self.input_folder.rglob(f'*{ext}'))
            image_files.extend(self.input_folder.rglob(f'*{ext.upper()}'))
        
        return sorted(image_files)
    
    def save_label(self, img_path, shape):
        """Save labeled image to appropriate folder"""
        dest_path = self.output_folder / shape / img_path.name
        
        # Handle duplicate filenames
        counter = 1
        original_name = img_path.name
        while dest_path.exists():
            stem = Path(original_name).stem
            ext = Path(original_name).suffix
            new_name = f"{stem}_{counter}{ext}"
            dest_path = self.output_folder / shape / new_name
            counter += 1
        
        shutil.copy(str(img_path), str(dest_path))
        return dest_path
    
    def run(self):
        """Main Streamlit app"""
        
        # Title and description
        st.title("👤 Face Shape Labeling Tool")
        st.markdown("---")
        
        # Initialize session state
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        if 'labeled_count' not in st.session_state:
            st.session_state.labeled_count = {shape: 0 for shape in self.shapes}
        if 'skipped_count' not in st.session_state:
            st.session_state.skipped_count = 0
        if 'image_files' not in st.session_state:
            st.session_state.image_files = self.get_image_files()
        
        image_files = st.session_state.image_files
        
        # Check if images exist
        if not image_files:
            st.error("❌ No images found in the input folder!")
            st.info(f"📁 Looking in: {self.input_folder}")
            return
        
        # Sidebar - Progress and Statistics
        with st.sidebar:
            st.header("📊 Progress")
            
            total_images = len(image_files)
            current_index = st.session_state.current_index
            total_labeled = sum(st.session_state.labeled_count.values())
            
            progress = current_index / total_images if total_images > 0 else 0
            st.progress(progress)
            
            st.metric("Current Image", f"{current_index + 1} / {total_images}")
            st.metric("Total Labeled", total_labeled)
            st.metric("Skipped", st.session_state.skipped_count)
            
            st.markdown("---")
            st.subheader("📈 Label Distribution")
            
            # Create bar chart
            df = pd.DataFrame({
                'Shape': list(st.session_state.labeled_count.keys()),
                'Count': list(st.session_state.labeled_count.values())
            })
            st.bar_chart(df.set_index('Shape'))
            
            st.markdown("---")
            
            # Detailed counts
            for shape, count in st.session_state.labeled_count.items():
                st.write(f"**{shape}:** {count}")
            
            st.markdown("---")
            
            # Reset button
            if st.button("🔄 Reset Progress", use_container_width=True):
                st.session_state.current_index = 0
                st.session_state.labeled_count = {shape: 0 for shape in self.shapes}
                st.session_state.skipped_count = 0
                st.rerun()
        
        # Check if all images are labeled
        if current_index >= total_images:
            st.success("🎉 All images have been labeled!")
            st.balloons()
            
            # Summary
            st.subheader("📊 Final Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Labeled", total_labeled)
            with col2:
                st.metric("Skipped", st.session_state.skipped_count)
            with col3:
                st.metric("Total Images", total_images)
            
            st.info(f"✅ Labeled images saved to: {self.output_folder}")
            return
        
        # Get current image
        current_img_path = image_files[current_index]
        
        # Display current image info
        col_info1, col_info2 = st.columns([3, 1])
        
        with col_info1:
            st.subheader(f"📷 Image: {current_img_path.name}")
            # Show original folder
            original_folder = current_img_path.parent.name
            if original_folder != 'train':
                st.caption(f"📂 Original folder: **{original_folder}**")
        
        with col_info2:
            st.metric("Progress", f"{current_index + 1}/{total_images}")
        
        st.markdown("---")
        
        # Main content area - Better organized layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display image with controlled size
            try:
                image = Image.open(str(current_img_path))
                
                # Resize image to a reasonable size (max 600px width)
                max_width = 600
                w, h = image.size
                if w > max_width:
                    ratio = max_width / w
                    new_h = int(h * ratio)
                    image = image.resize((max_width, new_h), Image.Resampling.LANCZOS)
                
                st.image(image, use_column_width=False)
                
            except Exception as e:
                st.error(f"❌ Could not load image: {e}")
                return
        
        with col2:
            st.markdown("### 🏷️ Select Face Shape")
            st.markdown("")
            
            # Create buttons for each shape - No emojis
            for i, shape in enumerate(self.shapes, 1):
                # Color coding for buttons
                button_type = "primary" if i == 1 else "secondary"
                
                if st.button(
                    f"{i}. {shape}", 
                    key=f"btn_{shape}",
                    use_container_width=True,
                    type=button_type if i == 1 else "secondary"
                ):
                    # Save the label
                    dest_path = self.save_label(current_img_path, shape)
                    st.session_state.labeled_count[shape] += 1
                    st.session_state.current_index += 1
                    
                    st.success(f"✅ Labeled as **{shape}**")
                    st.rerun()
            
            st.markdown("---")
            
            # Skip button
            if st.button("⏭️ Skip This Image", use_container_width=True, type="secondary"):
                st.session_state.skipped_count += 1
                st.session_state.current_index += 1
                st.info("⏭️ Image skipped")
                st.rerun()
            
            # Navigation
            st.markdown("---")
            st.markdown("### 🧭 Navigation")
            
            col_prev, col_next = st.columns(2)
            
            with col_prev:
                if st.button("⬅️ Prev", use_container_width=True, disabled=(current_index == 0)):
                    st.session_state.current_index = max(0, current_index - 1)
                    st.rerun()
            
            with col_next:
                if st.button("Next ➡️", use_container_width=True, disabled=(current_index >= total_images - 1)):
                    st.session_state.current_index = min(total_images - 1, current_index + 1)
                    st.rerun()
        
        # Quick stats at the bottom
        st.markdown("---")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        stats_cols = [col1, col2, col3, col4, col5]
        for i, shape in enumerate(self.shapes):
            with stats_cols[i]:
                st.metric(shape, st.session_state.labeled_count[shape])


if __name__ == "__main__":
    app = StreamlitLabellingApp()
    app.run()
