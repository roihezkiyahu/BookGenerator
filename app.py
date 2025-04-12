import streamlit as st
from main import build_graph
from tools.agent_tools import AgentState
import logging
import os
from typing import Dict, Any, Tuple, List, Optional, Union
from PIL import Image # Needed for checking image type and conversion
import io # Needed for image conversion

# Import utility functions
from app_utils import generate_random_inputs, create_pdf, sort_image_keys, process_image_for_pdf, generate_pdf_on_demand, generate_pptx_on_demand

# from dotenv import load_dotenv
# load_dotenv()

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Session State Initialization ---
def initialize_session_state():
    """Initializes necessary keys in Streamlit's session state."""
    defaults = {
        'character_count': 0,
        'page_index': 0,
        'merged_images': {},
        'page_texts': {},
        'title_input': "",
        'description_input': "",
        'style_input': "",
        'pages_input': 5,
        'is_generating': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for i in range(st.session_state.character_count):
        if f"char_name_{i}" not in st.session_state:
            st.session_state[f"char_name_{i}"] = ""
        if f"char_desc_{i}" not in st.session_state:
            st.session_state[f"char_desc_{i}"] = ""

# --- Character Input Management ---
def add_character():
    """Increments the character count and initializes new fields."""
    new_index = st.session_state.character_count
    st.session_state[f"char_name_{new_index}"] = ""
    st.session_state[f"char_desc_{new_index}"] = ""
    st.session_state.character_count += 1
    # Rerun needed to show new fields
    st.rerun()

def remove_character(index_to_remove: int):
    """Removes a character at a specific index and shifts subsequent characters."""
    current_count = st.session_state.character_count
    if 0 <= index_to_remove < current_count:
        for i in range(index_to_remove, current_count - 1):
            st.session_state[f"char_name_{i}"] = st.session_state.get(f"char_name_{i+1}", "")
            st.session_state[f"char_desc_{i}"] = st.session_state.get(f"char_desc_{i+1}", "")

        last_index = current_count - 1
        if f"char_name_{last_index}" in st.session_state:
            del st.session_state[f"char_name_{last_index}"]
        if f"char_desc_{last_index}" in st.session_state:
            del st.session_state[f"char_desc_{last_index}"]

        st.session_state.character_count -= 1
        if 'page_index' in st.session_state and st.session_state.page_index >= st.session_state.character_count:
             st.session_state.page_index = max(0, st.session_state.character_count - 1)
        # Rerun needed to reflect removal in UI
        st.rerun()
    else:
        logger.warning(f"Attempted to remove invalid character index: {index_to_remove}")

def display_character_inputs() -> List[Dict[str, str]]:
    """Displays input fields for characters vertically and collects their data."""
    st.subheader("Characters")
    characters_data = []
    # Don't process removals here, handle in button callback directly
    # indices_to_remove = []
    char_indices = list(range(st.session_state.character_count))

    for i in char_indices:
        col_title, col_button = st.columns([0.8, 0.2])
        with col_title:
            st.markdown(f"**Character {i+1}**")
        with col_button:
            # Use on_click for remove button, passing the index
            st.button(f"X", key=f"remove_char_{i}",
                      on_click=remove_character, args=(i,), # Pass index 'i' to the function
                      help=f"Remove Character {i+1}")

        char_name = st.text_input(
            "Name", key=f"char_name_{i}", label_visibility="collapsed",
            placeholder=f"Character {i+1} Name"
        )
        char_desc = st.text_area(
            "Description", key=f"char_desc_{i}", height=100,
            label_visibility="collapsed", placeholder=f"Character {i+1} Description"
        )
        characters_data.append({"name": char_name, "description": char_desc})
        if i < len(char_indices) - 1:
            st.divider()

    # Removed the block processing indices_to_remove

    st.button("Add Character", key="add_char_btn", on_click=add_character)
    return [char for char in characters_data if char.get("name")]

# --- Random Inputs Handling ---
def handle_randomize_click():
    """Calls the context-aware random input generator and updates session state intelligently."""
    logger.info("Randomize button clicked.")

    current_title = st.session_state.title_input
    current_description = st.session_state.description_input
    current_style = st.session_state.style_input
    current_pages = st.session_state.pages_input
    current_characters_list = []
    user_provided_chars = False
    for i in range(st.session_state.character_count):
        name = st.session_state.get(f"char_name_{i}")
        desc = st.session_state.get(f"char_desc_{i}")
        if name:
            current_characters_list.append({"name": name, "description": desc})
            user_provided_chars = True

    user_provided_title = bool(current_title)
    user_provided_desc = bool(current_description)
    user_provided_style = bool(current_style)

    random_data = generate_random_inputs(
        current_title=current_title or None,
        current_description=current_description or None,
        current_style=current_style or None,
        current_pages=current_pages,
        current_characters=current_characters_list if user_provided_chars else None
    )

    if random_data:
        logger.info(f"Merging random data: {random_data}")
        if not user_provided_title:
            st.session_state.title_input = random_data.get("title", current_title)
        if not user_provided_desc:
            st.session_state.description_input = random_data.get("story_description", current_description)
        if not user_provided_style:
            st.session_state.style_input = random_data.get("image_style", current_style)

        st.session_state.pages_input = random_data.get("number_of_pages", current_pages)

        if not user_provided_chars:
            generated_chars = random_data.get("characters", [])
            old_char_count = st.session_state.character_count
            st.session_state.character_count = len(generated_chars)

            for i in range(len(generated_chars)):
                 char = generated_chars[i]
                 if isinstance(char, dict):
                    st.session_state[f"char_name_{i}"] = char.get("name", "")
                    st.session_state[f"char_desc_{i}"] = char.get("description", "")
                 else:
                     logger.warning(f"Received invalid character format from LLM: {char}")
                     st.session_state[f"char_name_{i}"] = ""
                     st.session_state[f"char_desc_{i}"] = ""

            for i in range(len(generated_chars), old_char_count):
                if f"char_name_{i}" in st.session_state:
                    del st.session_state[f"char_name_{i}"]
                if f"char_desc_{i}" in st.session_state:
                    del st.session_state[f"char_desc_{i}"]

        logger.info("Session state updated with merged random inputs.")
        # No rerun needed here, handled by Streamlit
    else:
        logger.error("Failed to generate random inputs for merging.")
        st.error("Failed to generate random inputs. Please check the logs.")

# --- PDF Generation Logic ---
# Functions sort_image_keys, process_image_for_pdf moved to app_utils.py

def generate_pdf_for_download() -> Optional[bytes]:
    """Wrapper function to generate PDF using session state data.
    
    Returns:
        Optional[bytes]: The generated PDF bytes, or None if generation fails.
    """
    title = st.session_state.get('title_input', 'storybook')
    merged_images_data = st.session_state.get('merged_images', {})
    
    if not merged_images_data:
        logger.warning("PDF generation attempted but no images found in session state.")
        return None
        
    try:
        pdf_bytes = generate_pdf_on_demand(title, merged_images_data)
        if not pdf_bytes:
            st.error("Failed to generate PDF. Please try again.")
        return pdf_bytes
    except Exception as e:
        logger.exception(f"Error during PDF generation: {e}")
        st.error(f"Failed to generate PDF: {e}")
        return None

# --- Core Story Generation Logic ---
def generate_storybook(title: str, story_description: str, image_style: str,
                       number_of_pages: int, characters_metadata: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
    """Invokes the LangGraph agent to generate the storybook content."""
    logger.info(f"Starting storybook generation: '{title}' ({number_of_pages} pages)")
    characters_dict = {char['name']: char['description'] for char in characters_metadata if char.get('name')}

    try:
        graph = build_graph()
        initial_state = AgentState(
            title=title,
            number_of_pages=number_of_pages,
            story_description=story_description,
            image_style=image_style,
            characters_metadata=characters_dict,
            use_imagen_3=False,
            generated_images={},
            merged_images={},
            pages=[]
        )
        logger.debug(f"Invoking graph with initial state: {initial_state}")
        response = graph.invoke(initial_state)
        logger.info("Storybook generation graph invocation completed.")
        return response
    except Exception as e:
        logger.exception(f"Error during storybook generation graph invocation: {e}")
        st.error(f"An error occurred during generation: {e}")
        return None

def get_gallery_data() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Retrieves image and text data for the gallery from session state."""
    images_dict = st.session_state.get('merged_images', {})
    page_texts = st.session_state.get('page_texts', {})
    return images_dict, page_texts

def display_navigation_controls(page_keys: List[str]):
    """Displays Previous/Next buttons and the current page number/total."""
    total_pages = len(page_keys)
    if total_pages <= 1:
        return

    col1, col2, col3 = st.columns([1, 1, 1])
    current_index = st.session_state.page_index
    with col1:
        if st.button("Previous", key="prev_btn", use_container_width=True, disabled=(current_index == 0)):
            st.session_state.page_index -= 1
            st.rerun()
    with col2:
        try:
            current_page_num_str = page_keys[current_index].split('_')[-1]
            if current_index == 0:
                display_text = "Cover"
            else:
                display_text = f"Page {current_page_num_str} / {total_pages-1}"
        except:
            display_text = f"Page {current_index + 1} / {total_pages-1}"
        st.markdown(f"<p style='text-align: center; font-weight: bold;'>{display_text}</p>", unsafe_allow_html=True)
    with col3:
        if st.button("Next", key="next_btn", use_container_width=True, disabled=(current_index == total_pages - 1)):
            st.session_state.page_index += 1
            st.rerun()

def display_page_content(page_keys: List[str], images_dict: Dict[str, Any], page_texts: Dict[str, str]):
    """Displays the image (resized) and text for the currently selected page."""
    if not page_keys or st.session_state.page_index >= len(page_keys):
        logger.warning("Attempted to display page content with invalid index or empty keys.")
        return

    current_key = page_keys[st.session_state.page_index]
    is_cover = (st.session_state.page_index == 0)
    width_percentage = 75 if is_cover else 85

    if current_key in images_dict:
        img_data = images_dict[current_key]
        col_left, col_img, col_right = st.columns([
            (100 - width_percentage) / 2,
            width_percentage,
            (100 - width_percentage) / 2
        ])
        with col_img:
            st.image(img_data, use_container_width=True)
    else:
        st.warning(f"Image for page key '{current_key}' not found.")


def display_image_gallery():
    """Manages the display of the generated storybook gallery with navigation."""
    images_dict, page_texts = get_gallery_data()
    if not images_dict:
        return

    try:
        page_keys = sort_image_keys(list(images_dict.keys()))
    except ValueError as e:
        logger.error(f"Could not sort page keys: {e}")
        st.error("Error displaying gallery pages.")
        return

    if not page_keys:
        st.warning("No valid page keys found after sorting.")
        return

    if st.session_state.page_index >= len(page_keys) or st.session_state.page_index < 0:
        st.session_state.page_index = 0

    st.divider()
    st.header("Generated Storybook")
    if len(page_keys) > 0:
        try:
            current_page_num_str = page_keys[st.session_state.page_index].split('_')[-1]
            st.subheader(f"Page {current_page_num_str}")
        except:
             st.subheader(f"Page {st.session_state.page_index + 1}")

    display_navigation_controls(page_keys)
    st.divider()
    display_page_content(page_keys, images_dict, page_texts)

# --- Main Application Flow ---
def run_app():
    """Runs the main Streamlit application logic."""
    initialize_session_state()

    sidebar_state = "collapsed" if st.session_state.get('is_generating', False) else "expanded"
    st.set_page_config(layout="wide", initial_sidebar_state=sidebar_state)
    st.title("Storybook Generator")

    # --- Sidebar --- Always rendered
    with st.sidebar:
        st.header("Story Parameters")
        # Show inputs only if not generating
        if not st.session_state.get('is_generating', False):
            st.text_input("Title", key="title_input")
            st.text_area("Story Description", key="description_input", height=100)
            st.text_input("Image Style (e.g., Cartoon)", key="style_input")
            st.number_input("Number of Pages (3-9)", min_value=3, max_value=9, key="pages_input")
            st.button("Randomize Missing", on_click=handle_randomize_click, use_container_width=True,
                      help="Use AI to fill in any empty fields based on existing input.")
            characters_metadata_list = display_character_inputs()
            st.divider()

            # --- Generation Button ---
            if st.button("Generate Storybook", type="primary", use_container_width=True):
                gen_title = st.session_state.title_input
                gen_desc = st.session_state.description_input
                gen_style = st.session_state.style_input
                gen_pages = st.session_state.pages_input
                gen_chars = characters_metadata_list

                valid = True
                if not gen_title: st.warning("Please enter a Title."); valid = False
                if not gen_desc: st.warning("Please enter a Story Description."); valid = False
                if not gen_style: st.warning("Please enter an Image Style."); valid = False
                if not gen_chars: st.warning("Please add at least one Character."); valid = False

                if valid:
                    logger.info("Inputs valid, setting flag and caching inputs.")
                    st.session_state.gen_title_cache = gen_title
                    st.session_state.gen_desc_cache = gen_desc
                    st.session_state.gen_style_cache = gen_style
                    st.session_state.gen_pages_cache = gen_pages
                    st.session_state.gen_chars_cache = gen_chars
                    st.session_state.is_generating = True
                    # Immediately rerun to collapse sidebar and trigger generation phase
                    st.rerun()

            st.divider()
            # --- Export Options --- Rendered if images exist
            if st.session_state.get('merged_images'):
                st.write("**Export Options**")
                title_for_export = st.session_state.get('title_input', 'storybook')
                sanitized_title = title_for_export.replace(' ', '_').lower() or 'storybook'
                
                # Create columns for PDF and PPTX options
                col1, col2 = st.columns(2)
                
                # --- PDF Export --- 
                with col1:
                    st.write("PDF")
                    pdf_file_name = f"{sanitized_title}.pdf"
                    
                    # Handle PDF generation state
                    if 'pdf_data' not in st.session_state:
                        st.session_state.pdf_data = None
                    
                    if st.session_state.pdf_data is None:
                        if st.button("Generate PDF", use_container_width=True, key="generate_pdf_btn"):
                            with st.spinner("Generating PDF..."):
                                pdf_bytes = generate_pdf_for_download()
                                if pdf_bytes:
                                    st.session_state.pdf_data = pdf_bytes
                                    st.success("PDF generated successfully!")
                                else:
                                    st.error("Failed to generate PDF.")
                            st.rerun()
                    
                    if st.session_state.pdf_data is not None:
                        st.download_button(
                            label="Download PDF",
                            data=st.session_state.pdf_data,
                            file_name=pdf_file_name,
                            mime="application/pdf",
                            use_container_width=True,
                            key="pdf_download_button",
                            help="Download the generated storybook as a PDF file"
                        )
                        if st.button("Regenerate PDF", key="regenerate_pdf_btn", use_container_width=True):
                            st.session_state.pdf_data = None
                            st.rerun()
                
                # --- PPTX Export ---
                with col2:
                    st.write("PowerPoint")
                    pptx_file_name = f"{sanitized_title}.pptx"
                    
                    # Handle PPTX generation state
                    if 'pptx_data' not in st.session_state:
                        st.session_state.pptx_data = None
                    
                    if st.session_state.pptx_data is None:
                        if st.button("Generate PPTX", use_container_width=True, key="generate_pptx_btn"):
                            with st.spinner("Generating PowerPoint..."):
                                pptx_bytes = generate_pptx_on_demand(title_for_export, st.session_state.merged_images)
                                if pptx_bytes:
                                    st.session_state.pptx_data = pptx_bytes
                                    st.success("PowerPoint generated successfully!")
                                else:
                                    st.error("Failed to generate PowerPoint.")
                            st.rerun()
                    
                    if st.session_state.pptx_data is not None:
                        st.download_button(
                            label="Download PPTX",
                            data=st.session_state.pptx_data,
                            file_name=pptx_file_name,
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True,
                            key="pptx_download_button",
                            help="Download the generated storybook as a PowerPoint presentation"
                        )
                        if st.button("Regenerate PPTX", key="regenerate_pptx_btn", use_container_width=True):
                            st.session_state.pptx_data = None
                            st.rerun()
        else: # If is_generating is True
             st.info("Generation in progress...")

    # --- Generation Phase --- Handle if flag is set
    if st.session_state.get('is_generating', False):
        gen_title = st.session_state.get('gen_title_cache', "")
        gen_desc = st.session_state.get('gen_desc_cache', "")
        gen_style = st.session_state.get('gen_style_cache', "")
        gen_pages = st.session_state.get('gen_pages_cache', 5)
        gen_chars = st.session_state.get('gen_chars_cache', [])

        logger.info("Entering generation phase.")
        response = None
        with st.spinner("Generating your storybook..."):
            response = generate_storybook(gen_title, gen_desc, gen_style, gen_pages, gen_chars)

        processed_images = {}
        if response and response.get('merged_images'):
            logger.info(f"Generation successful.")
            # Attempt to convert images to PIL format for consistency
            for key, img_data in response['merged_images'].items():
                try:
                    if isinstance(img_data, Image.Image):
                        processed_images[key] = img_data
                    elif isinstance(img_data, bytes):
                        processed_images[key] = Image.open(io.BytesIO(img_data))
                    else:
                         logger.warning(f"Received unexpected image type ({type(img_data)}) for key {key}. Skipping conversion.")
                         processed_images[key] = img_data # Store as is if conversion fails
                except Exception as e:
                    logger.error(f"Failed to process/convert image for key {key}: {e}")
                    processed_images[key] = None # Or skip key

            st.session_state.merged_images = {k: v for k, v in processed_images.items() if v is not None}
            if 'pages' in response and isinstance(response['pages'], list):
                st.session_state.page_texts = {str(i+1): page_text for i, page_text in enumerate(response['pages'])}
            else:
                st.session_state.page_texts = {}
            st.session_state.page_index = 0
            st.success("Storybook generated successfully!")
        elif response:
            error_msg = response.get('error', "Generation completed, but no images found.")
            logger.warning(f"Generation response lacked images. Keys: {list(response.keys())}")
            st.error(error_msg)
            st.session_state.merged_images = {} # Clear any previous images
            st.session_state.page_texts = {}
        else:
            logger.error("Storybook generation failed.")
            st.error("Generation failed. Check logs.")
            st.session_state.merged_images = {} # Clear any previous images
            st.session_state.page_texts = {}

        st.session_state.is_generating = False
        # Clean up cached inputs
        for key in ['gen_title_cache', 'gen_desc_cache', 'gen_style_cache', 'gen_pages_cache', 'gen_chars_cache']:
             if key in st.session_state: del st.session_state[key]
        # Rerun to show results with expanded sidebar
        st.rerun()

    # --- Main Area Gallery Display --- Always check if images exist
    if st.session_state.get('merged_images'):
        display_image_gallery()
    # Show initial info only if not generating AND no images exist
    elif not st.session_state.get('is_generating', False):
        st.info("Enter story details in the sidebar, then click 'Generate Storybook'.")

if __name__ == "__main__":
    run_app()


