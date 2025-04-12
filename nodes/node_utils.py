import logging
from PIL import Image, ImageDraw, ImageFont
import os
import json
logger = logging.getLogger(__name__)

def get_font(font_size: int = 60):
    """
    Get an appropriate font for text rendering.

    Args:
        font_size: Size of the font to use.

    Returns:
        PIL.ImageFont.FreeTypeFont: The font to use for text rendering.
    """
    try:
        font = ImageFont.truetype("comic", font_size)
        logger.info("Using Comic Sans MS font (system)")
        return font
    except IOError:
        try:
            font = ImageFont.truetype("COMIC.TTF", font_size)
            logger.info("Using Comic Sans MS font (local .ttf)")
            return font
        except IOError:
            try:
                font = ImageFont.truetype("arial", font_size)
                logger.info("Using Arial font (system)")
                return font
            except IOError:
                try:
                    font = ImageFont.truetype("Arial.ttf", font_size)
                    logger.info("Using Arial font (local .ttf)")
                    return font
                except IOError:
                    logger.warning("All fonts unavailable. Using default font (fixed size).")
                    return ImageFont.load_default()

def calculate_text_width(text, font, draw):
    """Calculate the width of text with the given font.
    
    Args:
        text: The text to measure.
        font: The font to use.
        draw: The ImageDraw instance.
        
    Returns:
        int: The width of the text in pixels.
    """
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(text)
        return bbox[2] - bbox[0] if bbox else 0
    else:
        return draw.textlength(text, font=font)

def wrap_text(text, font, max_width, draw):
    """Wrap text to fit within a specified width.
    
    Args:
        text: The text to wrap.
        font: The font to use.
        max_width: Maximum width in pixels.
        draw: The ImageDraw instance.
        
    Returns:
        list: List of wrapped text lines.
    """
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if not current_line:
            current_line.append(word)
            continue
            
        test_line = ' '.join(current_line + [word])
        text_width = calculate_text_width(test_line, font, draw)
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def calculate_line_height(line, font):
    """Calculate the height of a line with the given font.
    
    Args:
        line: The text line to measure.
        font: The font to use.
        
    Returns:
        int: The height of the line in pixels.
    """
    if hasattr(font, "getbbox"):
        bbox = font.getbbox(line)
        return bbox[3] - bbox[1] if bbox else 72
    else:
        return getattr(font, "size", 72)

def check_text_overflow(y_position, line_height, image_height):
    """Check if text would overflow the image boundary.
    
    Args:
        y_position: Current vertical position.
        line_height: Height of the line.
        image_height: Total image height.
        
    Returns:
        bool: True if text would overflow, False otherwise.
    """
    if y_position + line_height > image_height - 30:
        logger.warning(f"Text truncated as it exceeds image height: {image_height}px")
        return True
    return False

def calculate_total_text_height(text: str, font, max_width: int, draw, image_height: int) -> int:
    """Calculate the total height needed to render the text.
    
    Args:
        text: The text to measure.
        font: The font to use.
        max_width: Maximum width in pixels.
        draw: The ImageDraw instance.
        image_height: Total image height.
        
    Returns:
        int: The total height needed for the text in pixels.
    """
    lines = wrap_text(text, font, max_width, draw)
    total_height = 30  # Starting y position
    
    for line in lines:
        line_height = calculate_line_height(line, font)
        total_height += line_height + 10  # Line height plus padding
    
    return total_height

def render_text_image(text: str, image_width: int, image_height: int, font=None, min_font_size: int = 12) -> Image.Image:
    """Create an image with rendered text, automatically adjusting font size if needed.
    
    Args:
        text: The text to render.
        image_width: Width of the image.
        image_height: Height of the image.
        font: The font to use. If None, a default font will be used.
        min_font_size: Minimum font size to try before giving up on fitting text.
        
    Returns:
        PIL.Image: Image with the rendered text.
    """
    text_frame_width = int(image_width *0.75)
    max_width = text_frame_width - 40
    font_size = 60  # Start with default font size
    
    # Try progressively smaller font sizes until text fits
    while font_size >= min_font_size:
        current_font = font if font is not None else get_font(font_size=font_size)
        
        # Create temporary image and draw for measurements
        temp_img = Image.new('RGB', (text_frame_width, image_height), color=(255, 249, 230))
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Calculate total height needed
        total_height = calculate_total_text_height(text, current_font, max_width, temp_draw, image_height)
        
        # If text fits, or we've reached minimum font size, use this font
        if total_height <= image_height - 30 or font_size <= min_font_size:
            logger.info(f"Using font size: {font_size}")
            break
            
        # Reduce font size and try again
        font_size -= 4
        logger.debug(f"Text too large, reducing font size to {font_size}")
    
    # Create the actual image with the final font size
    text_img = Image.new('RGB', (text_frame_width, image_height), color=(255, 249, 230))
    draw = ImageDraw.Draw(text_img)
    
    # Use the determined font
    final_font = font if font is not None else get_font(font_size=font_size)
    
    # Render the text
    lines = wrap_text(text, final_font, max_width, draw)
    y_position = 30
    
    for line in lines:
        # Calculate text width to center each line
        line_width = calculate_text_width(line, final_font, draw)
        x_position = (text_frame_width - line_width) // 2
        
        draw.text((x_position, y_position), line, fill='black', font=final_font)
        
        line_height = calculate_line_height(line, final_font)
        y_position += line_height + 10
        
        if check_text_overflow(y_position, line_height, image_height):
            # This should rarely happen now that we're adjusting font size
            logger.warning(f"Text still truncated with minimum font size {font_size}px")
            break
    
    return text_img

def create_merged_image_directory(title):
    """Create directory for merged images.
    
    Args:
        title: Book title used for directory name.
        
    Returns:
        str: Path to the created directory.
    """
    merged_dir = f"{title}/merged_images"
    os.makedirs(merged_dir, exist_ok=True)
    logger.info(f"Created directory for merged images: {merged_dir}")
    return merged_dir

def save_image(image, path):
    """Save image to specified path.
    
    Args:
        image: PIL.Image to save.
        path: Path where to save the image.
    """
    image.save(path)
    logger.info(f"Saved image: {path}")

def save_txt_file(text, path):
    """Save text to specified path.
    
    Args:
        text: Text to save.
        path: Path where to save the text.
    """
    with open(path, 'w') as f:
        json.dump(text, f)
    logger.info(f"Saved text: {path}")

__all__ = ['get_font', 'calculate_text_width', 'wrap_text', 'render_text_image', 
           'create_merged_image_directory', 'save_image', 'save_txt_file', 
           'calculate_line_height', 'check_text_overflow']
