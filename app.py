"""
Homes247 Premium Real Estate Image Processing - Streamlit Dashboard
Web-based Interface - Version 2.4 - WITH TEXT REMOVAL + WATERMARK
"""

# Suppress TensorFlow warnings FIRST
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import shutil
import uuid
from PIL import Image, ImageDraw, ImageFont
import cv2
import time
import zipfile
from io import BytesIO
import tempfile

# Suppress TensorFlow deprecation warnings
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Homes247 Premium - Image Processing",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== MODEL LOADING =====================
@st.cache_resource
def load_ai_model():
    BASE_DIR = r"C:\Users\Homes247\Desktop\Bulk_image"
    MODEL_PATH = os.path.join(BASE_DIR, 'classifier.h5')
    try:
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        CLASSES = ['floorplan', 'gallery', 'masterplan']
        return model, CLASSES, BASE_DIR
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, ['floorplan', 'gallery', 'masterplan'], None

model, CLASSES, BASE_DIR = load_ai_model()

# ===================== OCR LOADING FOR TEXT REMOVAL =====================
@st.cache_resource(show_spinner="Loading AI OCR engine for text removal...")
def load_ocr_reader():
    """Load EasyOCR reader for text detection and removal"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return reader
    except ImportError:
        st.warning("⚠️ EasyOCR not installed. Text removal will be skipped. Install with: pip install easyocr")
        return None
    except Exception as e:
        st.warning(f"⚠️ OCR loading failed: {e}. Text removal will be skipped.")
        return None

# Load OCR reader at startup
ocr_reader = load_ocr_reader()

# ===================== CONFIGURATION =====================
if BASE_DIR:
    OUTPUT_DIR = os.path.join(BASE_DIR, "api_output")
    UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
    TEMP_DIR = os.path.join(BASE_DIR, "api_temp")
    STATS_FILE = os.path.join(BASE_DIR, "api_processing_statistics.json")
    UPLOAD_HISTORY_FILE = os.path.join(BASE_DIR, "upload_history.json")
    WATERMARK_LOGO_FILE = os.path.join(BASE_DIR, "watermark_logo.png")
else:
    OUTPUT_DIR = "./api_output"
    UPLOAD_DIR = os.path.join(OUTPUT_DIR, "uploads")
    TEMP_DIR = "./api_temp"
    STATS_FILE = "./api_processing_statistics.json"
    UPLOAD_HISTORY_FILE = "./upload_history.json"
    WATERMARK_LOGO_FILE = "./watermark_logo.png"

CATEGORY_MAPPING = {
    "floorplan": "Floor Plan Images",
    "masterplan": "Master Plan Images",
    "gallery": "Gallery Images",
    "rejected": "Others"
}

CATEGORY_SIZES = {
    "floorplan": (1500, 1500),
    "masterplan": (1640, 860),
    "gallery": (820, 430),
    "rejected": None
}

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

for cat in ["floorplan", "masterplan", "gallery", "rejected"]:
    for quality in ["good_quality", "bad_quality"]:
        os.makedirs(os.path.join(OUTPUT_DIR, cat, quality), exist_ok=True)

# ===================== TEXT REMOVAL SETTINGS =====================
# Map internal categories to text removal categories
CATEGORY_TO_TEXT_REMOVAL = {
    "floorplan": "Floor Plan",
    "masterplan": "Master Plan",
    "gallery": "Gallery",
    "rejected": None  # Skip text removal for rejected images
}

# Keywords for text detection and removal
TEXT_REMOVAL_KEYWORDS = {
    "Master Plan": [
        'master', 'plan', 'legend', 'www', '.com', 'road', 'phase',
        'building', 'block', 'tower', 'future', 'extension', 'services',
        'north', 'entry', 'exit', 'copyright', 'reserved', 'logo',
        'trademark', 'developer', 'architect', 'scale', 'disclaimer',
    ],
    "Floor Plan": [
        'floor', 'plan', 'www', '.com', 'legend', 'scale', 'north',
        'copyright', 'reserved', 'logo', 'trademark', 'developer',
        'architect', 'disclaimer', 'note', 'not to scale',
    ],
    "Gallery": [
        'www', '.com', 'watermark', 'copyright', 'reserved', 'logo',
        'trademark', 'photo', 'image', 'stock', 'getty', 'shutterstock',
        'preview', 'sample', 'draft',
    ],
}

# Settings for text removal per category (AGGRESSIVE - Removes ALL edge text/logos)
TEXT_REMOVAL_SETTINGS = {
    "Master Plan": {"corner_pct": 0.18, "edge_pct": 0.10, "ocr_margin": 0.22},
    "Floor Plan": {"corner_pct": 0.15, "edge_pct": 0.08, "ocr_margin": 0.18},
    "Gallery": {"corner_pct": 0.12, "edge_pct": 0.06, "ocr_margin": 0.15},
}

# ===================== TEXT REMOVAL FUNCTIONS =====================
def build_removal_mask(img_bgr, category, ocr_reader, extra_margin=2, protect_center=True):
    """
    Build mask for text/logo removal - REMOVES ALL EDGE TEXT
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    cfg = TEXT_REMOVAL_SETTINGS.get(category, TEXT_REMOVAL_SETTINGS["Gallery"])
    keywords = TEXT_REMOVAL_KEYWORDS.get(category, TEXT_REMOVAL_KEYWORDS["Gallery"])
    
    ocr_margin = cfg["ocr_margin"] + extra_margin * 0.08
    
    # ── METHOD 1: OCR - REMOVE ALL TEXT AT EDGES ──────────────────────
    if ocr_reader is not None:
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            results = ocr_reader.readtext(gray)
            
            for det in results:
                bbox, text = det[0], det[1]
                pts = np.array(bbox, dtype=np.int32)
                cx = np.mean(pts[:, 0])
                cy = np.mean(pts[:, 1])
                
                # Check if at edge
                at_edge = (cx < w * ocr_margin or cx > w * (1 - ocr_margin) or
                          cy < h * ocr_margin or cy > h * (1 - ocr_margin))
                
                # REMOVE ALL TEXT AT EDGES (not just keywords)
                if at_edge:
                    exp = pts.copy()
                    exp[:, 0] = np.clip(pts[:, 0] + np.array([-8, 8, 8, -8]), 0, w)
                    exp[:, 1] = np.clip(pts[:, 1] + np.array([-8, -8, 8, 8]), 0, h)
                    cv2.fillPoly(mask, [exp.astype(np.int32)], 255)
        except Exception as e:
            pass
    
    # ── METHOD 2: Corner & edge blanking ────────────────────────────
    cx_pct = cfg["corner_pct"] + extra_margin * 0.03
    ey_pct = cfg["edge_pct"] + extra_margin * 0.03
    cxs = int(w * cx_pct)
    cys = int(h * cx_pct)
    es = int(min(h, w) * ey_pct)
    
    # Blank corners
    mask[0:cys, 0:cxs] = 255
    mask[0:cys, w-cxs:w] = 255
    mask[h-cys:h, 0:cxs] = 255
    mask[h-cys:h, w-cxs:w] = 255
    
    # Blank edges
    mask[0:es, :] = 255
    mask[h-es:h, :] = 255
    mask[:, 0:es] = 255
    mask[:, w-es:w] = 255
    
    # ── METHOD 3: Contour detection ────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = h * w
    edge_check = 0.18
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        near_edge = (x < w * edge_check or x + cw > w * (1 - edge_check) or
                    y < h * edge_check or y + ch > h * (1 - edge_check))
        
        is_small = area < total_area * 0.03
        aspect = max(cw, ch) / (min(cw, ch) + 1)
        is_text_shape = aspect > 3.0
        
        if near_edge and (is_small or is_text_shape):
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            xp = max(5, int(cw * 0.08))
            yp = max(5, int(ch * 0.08))
            cv2.rectangle(mask, 
                         (max(0, x - xp), max(0, y - yp)),
                         (min(w, x + cw + xp), min(h, y + ch + yp)),
                         255, -1)
    
    # ── METHOD 4: Legend box detection ─────────────────────────────
    if category in ("Master Plan", "Floor Plan"):
        edges_det = cv2.Canny(gray, 50, 150)
        kr = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        closed = cv2.morphologyEx(edges_det, cv2.MORPH_CLOSE, kr)
        rc, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in rc:
            x, y, cw, ch = cv2.boundingRect(cnt)
            a = cw * ch
            
            is_legend = 0.003 < (a / total_area) < 0.12
            
            at_edge_area = (x < w * 0.20 or x + cw > w * 0.80 or
                           y < h * 0.20 or y + ch > h * 0.80)
            
            asp = max(cw, ch) / (min(cw, ch) + 1)
            
            if is_legend and at_edge_area and 1.0 < asp < 4.0:
                cv2.rectangle(mask, (x - 8, y - 8), (x + cw + 8, y + ch + 8), 255, -1)
    
    # ── PROTECT CENTER ─────────────────────────────────────────────
    if category in ("Master Plan", "Floor Plan"):
        main_area = find_main_plan_area(img_bgr)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(main_area))
    
    # ── Finalize mask ──────────────────────────────────────────────
    k = np.ones((2, 2), np.uint8)
    mask = cv2.dilate(mask, k, iterations=1)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    return mask


def find_main_plan_area(img_bgr):
    """Find the main plan area to protect from removal - ENHANCED PROTECTION"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_mask = np.zeros((h, w), dtype=np.uint8)
    
    if contours:
        # Find larger contours (main plan areas) - REDUCED threshold to catch more
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            if cv2.contourArea(cnt) > h * w * 0.10:  # REDUCED from 0.15 - catch smaller protected areas
                cv2.drawContours(main_mask, [cnt], -1, 255, -1)
    
    # LARGER default protected area if no contours found
    if np.sum(main_mask) < h * w * 0.2:  # If protection is less than 20%
        # Protect larger center area (70% of image - INCREASED from 60%)
        cv2.rectangle(main_mask, 
                     (int(w * 0.15), int(h * 0.15)),  # INCREASED from 0.2
                     (int(w * 0.85), int(h * 0.85)),  # INCREASED from 0.8
                     255, -1)
    
    # LARGER dilation to expand protection
    kp = np.ones((15, 15), np.uint8)  # INCREASED from (10, 10)
    main_mask = cv2.dilate(main_mask, kp, iterations=2)  # INCREASED iterations from 1
    
    return main_mask


def remove_text_and_logos(image_path, output_path, category, ocr_reader, 
                          protect_center=True, extra_margin=2, inpaint_radius=3):
    """
    Remove text and logos from image - OPTIMIZED FOR QUALITY
    
    IMPROVEMENTS:
    - Prevents blur by using high-quality settings
    - Prevents cutting by conservative detection
    - Better edge preservation
    - Enhanced quality retention
    """
    try:
        # Load image in HIGHEST quality
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return False, 0.0, "Failed to read image"
        
        # Store original for quality comparison
        original_quality = cv2.Laplacian(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        
        # Build removal mask (with new conservative settings)
        mask = build_removal_mask(img_bgr, category, ocr_reader, extra_margin, protect_center)
        
        # Calculate percentage removed
        removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        
        # SAFETY CHECK: If removing too much, reduce mask
        if removed_pct > 25:  # If trying to remove more than 25%
            # Erode mask to reduce removal area
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=2)
            removed_pct = round(np.sum(mask == 255) / mask.size * 100, 2)
        
        # Apply inpainting ONLY if there's something to remove
        if np.sum(mask) > 0:
            # Use CONSERVATIVE inpaint radius
            safe_radius = min(inpaint_radius, 5)  # Cap at 5 to prevent blur
            
            # QUALITY-PRESERVING inpainting
            result = cv2.inpaint(img_bgr, mask, 
                               inpaintRadius=safe_radius,
                               flags=cv2.INPAINT_TELEA)
            
            # QUALITY CHECK: If result is too blurry, use less aggressive method
            result_quality = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            
            # If quality dropped significantly (>30%), try again with smaller radius
            if result_quality < original_quality * 0.7:
                result = cv2.inpaint(img_bgr, mask, 
                                   inpaintRadius=2,  # Use minimal radius
                                   flags=cv2.INPAINT_TELEA)
        else:
            result = img_bgr.copy()
        
        # Save with MAXIMUM quality to prevent compression blur
        cv2.imwrite(output_path, result, 
                   [cv2.IMWRITE_JPEG_QUALITY, 98,           # Increased from 95
                    cv2.IMWRITE_JPEG_OPTIMIZE, 1,           # Enable optimization
                    cv2.IMWRITE_JPEG_PROGRESSIVE, 0])       # Disable progressive (better quality)
        
        return True, removed_pct, None
        
    except Exception as e:
        return False, 0.0, str(e)


# ===================== WATERMARK MANAGEMENT =====================
def load_watermark_logo():
    """Load watermark logo from file"""
    if os.path.exists(WATERMARK_LOGO_FILE):
        try:
            logo = Image.open(WATERMARK_LOGO_FILE).convert("RGBA")
            return logo
        except Exception as e:
            st.error(f"Error loading watermark: {e}")
            return None
    return None


def save_watermark_logo(uploaded_file):
    """Save uploaded watermark logo"""
    try:
        logo = Image.open(uploaded_file).convert("RGBA")
        logo.save(WATERMARK_LOGO_FILE, "PNG")
        return True
    except Exception as e:
        st.error(f"Error saving watermark: {e}")
        return False


def apply_watermark_to_image(
    image_path,
    output_path,
    watermark_logo=None,
    logo_size_ratio=0.05,
    logo_opacity=0.20
):
    """
    Apply HIGH-QUALITY watermark logo to CENTER of image
    
    Parameters:
        - logo_size_ratio: float between 0.01–0.25 (recommended 0.05–0.12)
        - logo_opacity: float between 0.10–0.60 (recommended 0.20–0.40)
    """
    try:
        # Load main image in highest quality
        main_image = Image.open(image_path)
        if main_image.mode not in ('RGB', 'RGBA'):
            main_image = main_image.convert('RGB')
        img_width, img_height = main_image.size
        
        # Load watermark logo
        if watermark_logo is None:
            watermark_logo = load_watermark_logo()
        
        if watermark_logo is None:
            # No watermark available → save clean high-quality JPEG
            if main_image.mode == 'RGBA':
                main_image = main_image.convert('RGB')
            main_image.save(output_path, "JPEG", quality=100, subsampling=0, optimize=False)
            return True
        
        # Calculate logo size
        logo_width = int(img_width * logo_size_ratio)
        logo_width = max(logo_width, 40)  # minimum ~40px width
        logo_aspect_ratio = watermark_logo.height / watermark_logo.width
        logo_height = int(logo_width * logo_aspect_ratio)
        
        # Resize logo using highest quality resampling
        logo_resized = watermark_logo.resize(
            (logo_width, logo_height),
            Image.Resampling.LANCZOS
        )
        
        # Apply desired opacity
        if logo_resized.mode == 'RGBA':
            r, g, b, alpha = logo_resized.split()
            alpha = alpha.point(lambda p: int(p * logo_opacity))
            logo_resized = Image.merge('RGBA', (r, g, b, alpha))
        else:
            logo_resized = logo_resized.convert('RGBA')
            alpha = Image.new('L', logo_resized.size, int(255 * logo_opacity))
            logo_resized.putalpha(alpha)
        
        # Center position
        logo_x = (img_width - logo_width) // 2
        logo_y = (img_height - logo_height) // 2
        
        # Convert main image to RGBA for proper compositing
        if main_image.mode != 'RGBA':
            watermarked = main_image.convert('RGBA')
        else:
            watermarked = main_image.copy()
        
        # Paste logo using alpha compositing
        watermarked.paste(logo_resized, (logo_x, logo_y), logo_resized)
        
        # Convert back to RGB for JPEG saving
        watermarked_rgb = watermarked.convert('RGB')
        
        # Save with maximum quality settings
        watermarked_rgb.save(
            output_path,
            "JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
            progressive=False
        )
        
        return True
        
    except Exception as e:
        # Fallback: save original without watermark
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            img.save(output_path, "JPEG", quality=100, subsampling=0)
        except:
            pass
        return False


# ===================== UPLOAD HISTORY MANAGEMENT =====================
def load_upload_history():
    """Load upload history from JSON file"""
    if os.path.exists(UPLOAD_HISTORY_FILE):
        try:
            with open(UPLOAD_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"uploads": []}


def save_upload_history(history):
    """Save upload history to JSON file"""
    try:
        with open(UPLOAD_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        return True
    except Exception as e:
        return False


def add_upload_record(results):
    """Add a new upload record to history"""
    history = load_upload_history()
    
    new_floorplan = len([r for r in results if r.get('category_raw') == 'floorplan' and r.get('status') == 'success'])
    new_masterplan = len([r for r in results if r.get('category_raw') == 'masterplan' and r.get('status') == 'success'])
    new_gallery = len([r for r in results if r.get('category_raw') == 'gallery' and r.get('status') == 'success'])
    new_rejected = len([r for r in results if r.get('category_raw') == 'rejected' and r.get('status') == 'success'])
    new_good = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
    new_bad = len([r for r in results if r.get('quality_status') == 'Bad Quality' and r.get('status') == 'success'])
    
    successful = len([r for r in results if r.get('status') == 'success'])
    
    upload_record = {
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total": successful,
        "floorplan": new_floorplan,
        "masterplan": new_masterplan,
        "gallery": new_gallery,
        "rejected": new_rejected,
        "good_quality": new_good,
        "bad_quality": new_bad
    }
    
    history["uploads"].append(upload_record)
    save_upload_history(history)
    return upload_record


# ===================== STATISTICS MANAGEMENT =====================
def load_statistics():
    """Load statistics from JSON file"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "total_processed": 0,
        "floorplan_count": 0,
        "masterplan_count": 0,
        "gallery_count": 0,
        "rejected_count": 0,
        "good_quality_count": 0,
        "bad_quality_count": 0,
        "first_upload_date": None,
        "last_upload_date": None,
        "total_sessions": 0,
        "processing_history": []
    }


def save_statistics(stats):
    """Save statistics to JSON file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        return False


def update_statistics(results):
    """Update statistics with new processing results"""
    stats = load_statistics()
    
    new_floorplan = len([r for r in results if r.get('category_raw') == 'floorplan' and r.get('status') == 'success'])
    new_masterplan = len([r for r in results if r.get('category_raw') == 'masterplan' and r.get('status') == 'success'])
    new_gallery = len([r for r in results if r.get('category_raw') == 'gallery' and r.get('status') == 'success'])
    new_rejected = len([r for r in results if r.get('category_raw') == 'rejected' and r.get('status') == 'success'])
    new_good = len([r for r in results if r.get('quality_status') == 'Good Quality' and r.get('status') == 'success'])
    new_bad = len([r for r in results if r.get('quality_status') == 'Bad Quality' and r.get('status') == 'success'])
    
    total_new = new_floorplan + new_masterplan + new_gallery + new_rejected
    
    stats['total_processed'] += total_new
    stats['floorplan_count'] += new_floorplan
    stats['masterplan_count'] += new_masterplan
    stats['gallery_count'] += new_gallery
    stats['rejected_count'] += new_rejected
    stats['good_quality_count'] += new_good
    stats['bad_quality_count'] += new_bad
    
    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if stats['first_upload_date'] is None:
        stats['first_upload_date'] = current_date
    stats['last_upload_date'] = current_date
    stats['total_sessions'] += 1
    
    stats['processing_history'].append({
        'date': current_date,
        'total': total_new,
        'floorplan': new_floorplan,
        'masterplan': new_masterplan,
        'gallery': new_gallery,
        'rejected': new_rejected,
        'good_quality': new_good,
        'bad_quality': new_bad
    })
    
    if len(stats['processing_history']) > 100:
        stats['processing_history'] = stats['processing_history'][-100:]
    
    save_statistics(stats)
    return stats


def format_file_size(size_bytes):
    """Format bytes to human readable format"""
    if size_bytes is None:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


# ===================== PREMIUM DARK THEME CSS =====================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #1e0f32 0%, #0f0520 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e0e7ff !important;
        font-weight: 700;
    }
    
    /* Cards and containers */
    .stAlert, .stExpander {
        background: rgba(30, 15, 50, 0.6);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 8px 20px rgba(139, 92, 246, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #8b5cf6;
        font-size: 2rem;
        font-weight: 800;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
    }
</style>
""", unsafe_allow_html=True)


# ===================== UTILITY FUNCTIONS =====================
def predict_image(img_path: str):
    if model is None:
        return 'error', 0.0, {}
    try:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        preds = model.predict(arr, verbose=0)[0]
        idx = np.argmax(preds)
        label = CLASSES[idx]
        confidence = float(preds[idx])
        all_probs = {CLASSES[i]: float(preds[i]) for i in range(len(CLASSES))}
        return label, confidence, all_probs
    except Exception as e:
        return 'error', 0.0, {}


def assess_image_quality(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0, {"error": "Cannot read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = min(100, (laplacian.var() / 500) * 100)
        
        brightness = np.mean(gray)
        brightness_score = 100 - (abs(127 - brightness) / 127 * 100)
        
        contrast = np.std(gray)
        contrast_score = min(100, (contrast / 64) * 100)
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        blur_score = min(100, np.mean(magnitude_spectrum) / 2)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_score = min(100, (np.sum(edges > 0) / edges.size) * 500)
        
        quality_score = (
            sharpness_score * 0.35 +
            brightness_score * 0.15 +
            contrast_score * 0.20 +
            blur_score * 0.15 +
            edge_score * 0.15
        )
        
        return quality_score, {
            "overall": round(quality_score, 2),
            "sharpness": round(sharpness_score, 2),
            "brightness": round(brightness_score, 2),
            "contrast": round(contrast_score, 2),
            "blur": round(blur_score, 2),
            "edge": round(edge_score, 2)
        }
    except Exception as e:
        return 0, {"error": f"Processing failed: {str(e)}"}


def resize_image(image_path, target_size, output_path, watermark_logo=None):
    """Resize image (watermark will be applied separately)"""
    try:
        img = Image.open(image_path).convert('RGB')
        orig_w, orig_h = img.size
        target_w, target_h = target_size
        
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
        
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        final_img = Image.new('RGB', target_size, (255, 255, 255))
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        final_img.paste(img_resized, (paste_x, paste_y))
        
        # Save resized image without watermark
        final_img.save(output_path, 'JPEG', quality=95)
        
        return True
    except Exception as e:
        return False


def process_single_image(image_path, filename, conf_thresh, qual_thresh, file_size=None, 
                        watermark_logo=None, enable_text_removal=True, text_removal_settings=None):
    """
    PROCESSING PIPELINE:
    1. AI Classification
    2. Quality Analysis
    3. Auto-Resize (if needed)
    4. AI Text & Logo Removal (NEW - for non-rejected images)
    5. Watermark Application (ALL IMAGES)
    6. Save to Output
    """
    try:
        # STEP 1: AI Classification
        label, conf, all_probs = predict_image(image_path)
        
        # STEP 2: Quality Analysis
        quality_score, metrics = assess_image_quality(image_path)
        
        category = "rejected" if conf < conf_thresh else label
        quality_status = "Good Quality" if quality_score >= qual_thresh else "Bad Quality"
        
        with Image.open(image_path) as img:
            width, height = img.size
        
        quality_folder = "good_quality" if quality_score >= qual_thresh else "bad_quality"
        output_dir = os.path.join(OUTPUT_DIR, category, quality_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        base, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base}.jpg")
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{base}_{counter}.jpg")
            counter += 1
        
        resize_info = "Original"
        text_removal_info = "Skipped"
        text_removal_pct = 0.0
        
        # Use temp files for multi-step processing
        temp_resized = None
        temp_cleaned = None
        
        # STEP 3: Resize (if not rejected and size specified)
        if category != "rejected" and CATEGORY_SIZES[category]:
            temp_resized = os.path.join(TEMP_DIR, f"temp_resized_{uuid.uuid4().hex}.jpg")
            if resize_image(image_path, CATEGORY_SIZES[category], temp_resized, watermark_logo):
                resize_info = f"{CATEGORY_SIZES[category][0]}×{CATEGORY_SIZES[category][1]}"
                current_file = temp_resized
            else:
                resize_info = "Resize Failed"
                current_file = image_path
        else:
            current_file = image_path
        
        # STEP 4: AI Text & Logo Removal (NEW - only for non-rejected images)
        if enable_text_removal and category != "rejected" and ocr_reader is not None:
            text_category = CATEGORY_TO_TEXT_REMOVAL.get(category)
            if text_category:
                temp_cleaned = os.path.join(TEMP_DIR, f"temp_cleaned_{uuid.uuid4().hex}.jpg")
                
                # Get text removal settings
                tr_settings = text_removal_settings or {}
                protect_center = tr_settings.get('protect_center', True)
                extra_margin = tr_settings.get('extra_margin', 2)
                inpaint_radius = tr_settings.get('inpaint_radius', 3)
                
                success, removed_pct, error = remove_text_and_logos(
                    current_file, 
                    temp_cleaned, 
                    text_category,
                    ocr_reader,
                    protect_center,
                    extra_margin,
                    inpaint_radius
                )
                
                if success:
                    text_removal_info = f"Removed {removed_pct}% text/logos"
                    text_removal_pct = removed_pct
                    current_file = temp_cleaned
                else:
                    text_removal_info = f"Failed: {error}" if error else "Failed"
        
        # STEP 5: Watermark Application (to final file)
        apply_watermark_to_image(current_file, output_path, watermark_logo)
        
        # Clean up temp files
        if temp_resized and os.path.exists(temp_resized):
            os.remove(temp_resized)
        if temp_cleaned and os.path.exists(temp_cleaned):
            os.remove(temp_cleaned)
        
        return {
            "filename": filename,
            "file_size": format_file_size(file_size) if file_size else "N/A",
            "category": CATEGORY_MAPPING.get(category, category),
            "category_raw": category,
            "confidence": round(conf * 100, 2),
            "all_probabilities": {k: round(v * 100, 2) for k, v in all_probs.items()},
            "quality_status": quality_status,
            "quality_score": round(quality_score, 2),
            "sharpness": metrics.get("sharpness", 0),
            "brightness": metrics.get("brightness", 0),
            "contrast": metrics.get("contrast", 0),
            "blur": metrics.get("blur", 0),
            "edge": metrics.get("edge", 0),
            "width": width,
            "height": height,
            "resolution": f"{width}×{height}",
            "output_size": resize_info,
            "text_removal": text_removal_info,
            "text_removal_pct": text_removal_pct,
            "output_path": output_path,
            "quality_folder": quality_folder,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "filename": filename,
            "file_size": format_file_size(file_size) if file_size else "N/A",
            "category": "Error",
            "category_raw": "error",
            "confidence": 0,
            "quality_status": "Error",
            "quality_score": 0,
            "text_removal": "Error",
            "text_removal_pct": 0,
            "status": "failed",
            "error": str(e)
        }


def create_download_zip(results: list, timestamp: str):
    zip_buffer = BytesIO()
    
    try:
        df = pd.DataFrame(results)
        
        total = len(results)
        successful = len([r for r in results if r.get('status') == 'success'])
        property_img = len([r for r in results if r.get('category_raw') == 'gallery'])
        floor_img = len([r for r in results if r.get('category_raw') == 'floorplan'])
        master_img = len([r for r in results if r.get('category_raw') == 'masterplan'])
        rejected_img = len([r for r in results if r.get('category_raw') == 'rejected'])
        good_qual = len([r for r in results if r.get('quality_status') == 'Good Quality'])
        bad_qual = len([r for r in results if r.get('quality_status') == 'Bad Quality'])
        
        df_success = df[df['status'] == 'success']
        avg_conf = df_success['confidence'].mean() if len(df_success) > 0 else 0
        avg_qual = df_success['quality_score'].mean() if len(df_success) > 0 else 0
        avg_text_removal = df_success['text_removal_pct'].mean() if len(df_success) > 0 else 0
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(f"reports/homes247_report_{timestamp}.csv", df.to_csv(index=False))
            zip_file.writestr(f"reports/homes247_report_{timestamp}.json", json.dumps(results, indent=2))
            
            images_added = 0
            for result in results:
                if result.get('status') == 'success' and 'output_path' in result:
                    output_path = result['output_path']
                    if os.path.exists(output_path):
                        category = result['category_raw']
                        quality_folder = result.get('quality_folder', 'unknown')
                        zip_path = f"images/{category}/{quality_folder}/{os.path.basename(output_path)}"
                        with open(output_path, 'rb') as img_file:
                            zip_file.writestr(zip_path, img_file.read())
                        images_added += 1
            
            summary_text = f"""HOMES247 - PROCESSING SUMMARY
=========================================
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {timestamp}

STATISTICS
----------
Total Images: {total}
Successfully Processed: {successful} ({successful/total*100:.1f}%)
Failed: {total-successful}
Images in ZIP: {images_added}

CATEGORIES
----------
Gallery: {property_img}
Floor Plans: {floor_img}
Master Plans: {master_img}
Rejected: {rejected_img}

QUALITY
-------
Good Quality: {good_qual} ({good_qual/total*100:.1f}%)
Bad Quality: {bad_qual} ({bad_qual/total*100:.1f}%)

METRICS
-------
Avg Confidence: {avg_conf:.2f}%
Avg Quality: {avg_qual:.2f}%
Avg Text Removed: {avg_text_removal:.2f}%

TEXT & LOGO REMOVAL
-------------------
✓ AI-powered OCR detection
✓ Corner & edge blanking
✓ Legend box removal
✓ Smart inpainting
✓ Center content protection

WATERMARK
---------
✓ Applied to ALL images
✓ Position: Center
✓ Opacity: 20% (professional)

OUTPUT SIZES
------------
Floor Plans: 1500×1500
Master Plans: 1640×860
Gallery: 820×430
Rejected: Original

---
Homes247 - India's Favourite Property Portal
Streamlit Dashboard Version 2.4
"""
            zip_file.writestr("README.txt", summary_text)
        
        zip_buffer.seek(0)
        return zip_buffer
    except Exception as e:
        st.error(f"ZIP creation failed: {str(e)}")
        return None


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 10px; margin-bottom: 1rem;'>
    <h2 style='margin: 0; color: white;'>⚙️ Configuration</h2>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Watermark Logo Upload
    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #8b5cf6;'>🏠 Watermark Logo</h3>
    <p style='margin: 0; font-size: 0.9rem; color: #c4b5fd;'>Upload your logo for automatic watermarking</p>
</div>
""", unsafe_allow_html=True)
    
    watermark_upload = st.file_uploader(
        "Upload Logo (PNG recommended)",
        type=['png', 'jpg', 'jpeg'],
        key="watermark_uploader",
        help="Logo will be applied to center at 20% opacity"
    )
    
    if watermark_upload:
        if save_watermark_logo(watermark_upload):
            st.success("✅ Watermark logo saved!")
            st.rerun()
    
    # Show watermark status
    current_logo = load_watermark_logo()
    if current_logo:
        st.info("✓ Watermark: ACTIVE")
        st.image(current_logo, width=100, caption="Current Logo")
        if st.button("🗑️ Remove Watermark", use_container_width=True):
            if os.path.exists(WATERMARK_LOGO_FILE):
                os.remove(WATERMARK_LOGO_FILE)
                st.success("Watermark removed!")
                st.rerun()
    else:
        st.warning("⚠️ No watermark logo uploaded")
    
    st.markdown("---")
    
    # Text Removal Settings
    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #8b5cf6;'>🧹 Text Removal</h3>
    <p style='margin: 0; font-size: 0.9rem; color: #c4b5fd;'>AI-powered text & logo removal</p>
</div>
""", unsafe_allow_html=True)
    
    enable_text_removal = st.checkbox(
        "🤖 Enable Text & Logo Removal",
        value=True,
        help="Remove text, logos, legends from images using AI"
    )
    
    if enable_text_removal:
        if ocr_reader is not None:
            st.success("✅ OCR Engine Ready")
        else:
            st.error("❌ OCR not available")
        
        protect_center = st.checkbox(
            "🛡️ Protect Central Content",
            value=True,
            help="Prevent removal of main plan content (HIGHLY RECOMMENDED)"
        )
        
        extra_margin = st.slider(
            "🔍 Detection Aggressiveness",
            0, 5, 2,  # DEFAULT: 2 (balanced)
            help="Higher = removes more. 2 = balanced removal."
        )
        
        inpaint_radius = st.slider(
            "🎨 Inpaint Smoothness",
            1, 8, 3,  # DEFAULT: 3 (smooth)
            help="Smoothness for filling removed areas."
        )
        
        if extra_margin > 3:
            st.warning("⚠️ Aggressiveness >3 may affect image quality.")
        
        if inpaint_radius > 5:
            st.warning("⚠️ Smoothness >5 may cause slight blur.")
            
    else:
        protect_center = True
        extra_margin = 2
        inpaint_radius = 3
    
    st.markdown("---")
    
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        0, 100, 70, 5,
        help="Minimum confidence to accept classification"
    )
    
    quality_threshold = st.slider(
        "🌟 Quality Threshold",
        0, 100, 50, 5,
        help="Minimum quality score for 'Good Quality'"
    )
    
    st.markdown("---")
    
    # Unlimited processing info
    st.markdown("""
<div style='text-align: center; background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 1rem; border-radius: 10px;'>
    <h3 style='margin: 0; color: white;'>∞ UNLIMITED ∞</h3>
    <p style='margin: 0.5rem 0 0 0; color: #d1fae5;'>✓ Unlimited Images</p>
    <p style='margin: 0; color: #d1fae5;'>✓ Unlimited File Size</p>
    <p style='margin: 0; color: #d1fae5; font-weight: bold;'>No restrictions!</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
    <h3 style='margin: 0 0 1rem 0; color: #8b5cf6;'>📋 Processing Pipeline</h3>
    <div style='color: #c4b5fd;'>
        1️⃣ Upload Images<br/>
        2️⃣ AI Classification<br/>
        3️⃣ Quality Analysis<br/>
        4️⃣ Auto-Resize<br/>
        <span style='color: #10b981; font-weight: bold;'>4.5️⃣ AI Text & Logo Removal ✨ NEW</span><br/>
        5️⃣ Watermark (ALL IMAGES)<br/>
        6️⃣ Organize & Save
    </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
    <h3 style='margin: 0 0 0.5rem 0; color: #8b5cf6;'>📐 Output Sizes</h3>
    <div style='color: #c4b5fd; font-size: 0.9rem;'>
        🏠 Floor Plans: 1500×1500<br/>
        🗺️ Master Plans: 1640×860<br/>
        🖼️ Gallery: 820×430<br/>
        ❌ Rejected: Original
    </div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Reset statistics button
    if st.button("🔄 Reset All Statistics", use_container_width=True):
        if os.path.exists(STATS_FILE):
            os.remove(STATS_FILE)
        if os.path.exists(UPLOAD_HISTORY_FILE):
            os.remove(UPLOAD_HISTORY_FILE)
        st.success("Statistics reset successfully!")
        st.rerun()


# ===================== MAIN HEADER =====================
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 15px; margin-bottom: 2rem;'>
    <h1 style='margin: 0; color: white; font-size: 3rem;'>🏠 HOMES247</h1>
    <p style='margin: 0.5rem 0 0 0; color: #e9d5ff; font-size: 1.2rem;'>India's Favourite Property Portal - AI Dashboard</p>
    <p style='margin: 0.5rem 0 0 0; color: #fae8ff; font-size: 0.9rem;'>∞ UNLIMITED UPLOAD-IMAGES ✓ QUALITY-CHECK ∞ AUTO RESIZE ✓ <span style='color: #10b981;'>AI TEXT REMOVAL ✨</span> ✓ AUTO WATERMARK</p>
</div>
""", unsafe_allow_html=True)


# ===================== PERSISTENT STATISTICS BANNER =====================
stats = load_statistics()

if stats['total_processed'] > 0:
    st.markdown(f"""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0 0 1rem 0;'>📊 ALL-TIME PROCESSING STATISTICS</h2>
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;'>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['floorplan_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>🏠 Floor Plans</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['masterplan_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>🗺️ Master Plans</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['gallery_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>🖼️ Gallery</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['total_processed']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>📊 Total Processed</div>
        </div>
    </div>
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;'>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #ef4444; font-weight: bold;'>{stats['rejected_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>❌ Rejected</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #10b981; font-weight: bold;'>{stats['good_quality_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>✅ Good Quality</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #f59e0b; font-weight: bold;'>{stats['bad_quality_count']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>⚠️ Bad Quality</div>
        </div>
        <div style='text-align: center; background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px;'>
            <div style='font-size: 2rem; color: #8b5cf6; font-weight: bold;'>{stats['total_sessions']:,}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>🔢 Sessions</div>
        </div>
    </div>
    <div style='text-align: center; margin-top: 1rem; color: #c4b5fd; font-size: 0.9rem;'>
        📅 First Upload: {stats['first_upload_date'] if stats['first_upload_date'] else 'N/A'}   |   🕐 Last Upload: {stats['last_upload_date'] if stats['last_upload_date'] else 'N/A'}
    </div>
</div>
""", unsafe_allow_html=True)


# ===================== SESSION STATE =====================
if 'results' not in st.session_state:
    st.session_state.results = []
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'processing' not in st.session_state:
    st.session_state.processing = False


# ===================== UPLOAD SECTION =====================
st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📤 Upload Real Estate Images - COMPLETELY UNLIMITED</h2>
</div>
""", unsafe_allow_html=True)

# Unlimited info banner
st.markdown("""
<div style='background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3 style='text-align: center; color: white; margin: 0 0 0.5rem 0;'>⚡ NO LIMITS - PROCESS ANY NUMBER OF IMAGES OF ANY SIZE ⚡</h3>
    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;'>
        <div style='text-align: center; background: rgba(255, 255, 255, 0.1); padding: 0.75rem; border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>∞</div>
            <div style='font-weight: bold; color: white;'>Unlimited Quantity</div>
            <div style='color: #d1fae5; font-size: 0.9rem;'>Upload 1000+ images!</div>
        </div>
        <div style='text-align: center; background: rgba(255, 255, 255, 0.1); padding: 0.75rem; border-radius: 8px;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>∞</div>
            <div style='font-weight: bold; color: white;'>Unlimited File Size</div>
            <div style='color: #d1fae5; font-size: 0.9rem;'>From KB to GB+!</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag & Drop or Browse - Upload ANY number of images with ANY file size",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=True,
    help="UNLIMITED: Upload as many images as you need with any file size"
)

if uploaded_files:
    total_images = len(uploaded_files)
    total_size = sum(file.size for file in uploaded_files)
    
    # Display upload statistics
    st.markdown(f"""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    <h3 style='text-align: center; color: #8b5cf6; margin: 0 0 1rem 0;'>📊 Upload Statistics</h3>
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
        <div style='text-align: center;'>
            <div style='font-size: 1.5rem; color: #8b5cf6; font-weight: bold;'>{total_images}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>Total Images</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 1.5rem; color: #8b5cf6; font-weight: bold;'>{format_file_size(total_size)}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>Total Size</div>
        </div>
        <div style='text-align: center;'>
            <div style='font-size: 1.5rem; color: #8b5cf6; font-weight: bold;'>{format_file_size(total_size/total_images)}</div>
            <div style='color: #c4b5fd; font-size: 0.9rem;'>Average Size</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.success(f"✅ **{total_images} images** ({format_file_size(total_size)}) ready for processing!")
        
        # Show estimate
        estimated_time = total_images * 2
        st.info(f"⏱️ Estimated processing time: ~{estimated_time // 60} min {estimated_time % 60} sec")
    
    if st.button("🚀 PROCESS ALL IMAGES", use_container_width=True, disabled=st.session_state.processing):
        st.session_state.results = []
        st.session_state.processed = False
        st.session_state.processing = True
        
        # Load watermark logo once
        watermark_logo = load_watermark_logo()
        
        # Prepare text removal settings
        text_removal_settings = {
            'protect_center': protect_center,
            'extra_margin': extra_margin,
            'inpaint_radius': inpaint_radius
        }
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_container = st.container()
        
        # Processing metrics
        start_time = time.time()
        success_count = 0
        failed_count = 0
        total_processed_size = 0
        
        # Status display
        with status_container:
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            metric_total = col_stat1.empty()
            metric_success = col_stat2.empty()
            metric_failed = col_stat3.empty()
            metric_size = col_stat4.empty()
            status_text = st.empty()
        
        # Process all images without limit
        for idx, file in enumerate(uploaded_files):
            current_num = idx + 1
            file_size = file.size
            total_processed_size += file_size
            
            # Update status
            status_text.markdown(f"""
<div style='background: rgba(139, 92, 246, 0.1); padding: 1rem; border-radius: 8px; text-align: center;'>
    <h3 style='color: #8b5cf6; margin: 0 0 0.5rem 0;'>⏳ Processing Image {current_num}/{total_images}</h3>
    <p style='margin: 0; color: #c4b5fd;'>File: {file.name}</p>
    <p style='margin: 0; color: #c4b5fd;'>Size: {format_file_size(file_size)}</p>
    <p style='margin: 0.5rem 0 0 0; color: #8b5cf6; font-weight: bold;'>Progress: {(current_num/total_images)*100:.1f}%</p>
</div>
""", unsafe_allow_html=True)
            
            # Update metrics
            metric_total.metric("📊 Processed", f"{current_num}/{total_images}")
            metric_success.metric("✅ Success", success_count)
            metric_failed.metric("❌ Failed", failed_count)
            metric_size.metric("💾 Processed", format_file_size(total_processed_size))
            
            try:
                # Save uploaded file
                temp_path = os.path.join(UPLOAD_DIR, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # Process image with text removal and watermark
                result = process_single_image(
                    temp_path,
                    file.name,
                    confidence_threshold / 100,
                    quality_threshold,
                    file_size,
                    watermark_logo,
                    enable_text_removal,
                    text_removal_settings
                )
                
                st.session_state.results.append(result)
                
                if result['status'] == 'success':
                    success_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                st.session_state.results.append({
                    "filename": file.name,
                    "file_size": format_file_size(file_size),
                    "category": "Error",
                    "category_raw": "error",
                    "confidence": 0,
                    "quality_status": "Error",
                    "quality_score": 0,
                    "sharpness": 0,
                    "brightness": 0,
                    "contrast": 0,
                    "resolution": "N/A",
                    "output_size": "N/A",
                    "text_removal": "Error",
                    "text_removal_pct": 0,
                    "status": "failed",
                    "error": str(e)
                })
            
            # Update progress bar
            progress_bar.progress(current_num / total_images)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update persistent statistics
        updated_stats = update_statistics(st.session_state.results)
        
        # Add upload record to history
        upload_record = add_upload_record(st.session_state.results)
        
        # Final status
        st.session_state.processed = True
        st.session_state.processing = False
        status_text.empty()
        progress_bar.empty()
        
        # Show completion message
        st.balloons()
        
        # Calculate text removal stats
        df_results = pd.DataFrame(st.session_state.results)
        df_success = df_results[df_results['status'] == 'success']
        avg_text_removal = df_success['text_removal_pct'].mean() if len(df_success) > 0 else 0
        
        st.success(f"""
🎉 **Processing Complete!**

- Total Images: **{total_images}**
- Total Data: **{format_file_size(total_size)}**
- Successfully Processed: **{success_count}**
- Failed: **{failed_count}**
- Time Taken: **{processing_time//60:.0f} min {processing_time%60:.0f} sec**
- Average Time/Image: **{processing_time/total_images:.2f} sec**
- Processing Speed: **{format_file_size(total_size/processing_time)}/sec**

🧹 **AI Text Removal:** {"Enabled" if enable_text_removal and ocr_reader else "Disabled"} {"(Avg: " + f"{avg_text_removal:.1f}% removed)" if enable_text_removal and ocr_reader and avg_text_removal > 0 else ""}

🏠 **Watermark Applied:** {"ALL images" if watermark_logo else "No watermark (upload logo in sidebar)"}

📊 **Statistics Updated!** Refresh the page to see updated all-time counts.

📝 **Upload Record Saved:** {upload_record.get('date', 'N/A')}
        """)
        
        # Auto-refresh to show updated statistics
        time.sleep(2)
        st.rerun()


# ===================== RESULTS DISPLAY =====================
if st.session_state.processed and st.session_state.results:
    df = pd.DataFrame(st.session_state.results)
    
    # METRICS
    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin-bottom: 2rem;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📊 Current Session Analytics</h2>
</div>
""", unsafe_allow_html=True)
    
    total = len(df)
    successful = len(df[df['status'] == 'success'])
    property_img = len(df[df['category_raw'] == 'gallery'])
    floor_img = len(df[df['category_raw'] == 'floorplan'])
    master_img = len(df[df['category_raw'] == 'masterplan'])
    rejected_img = len(df[df['category_raw'] == 'rejected'])
    good_qual = len(df[df['quality_status'] == 'Good Quality'])
    bad_qual = len(df[df['quality_status'] == 'Bad Quality'])
    
    # Calculate averages only for successful images
    df_success = df[df['status'] == 'success']
    avg_conf = df_success['confidence'].mean() if len(df_success) > 0 else 0
    avg_qual = df_success['quality_score'].mean() if len(df_success) > 0 else 0
    avg_text_removal = df_success['text_removal_pct'].mean() if len(df_success) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🗺️ Master Plans", master_img)
        st.metric("✅ Successful", successful, f"{successful/total*100:.1f}%")
    
    with col2:
        st.metric("🖼️ Gallery Images", property_img)
        st.metric("❌ Rejected", rejected_img)
    
    with col3:
        st.metric("📐 Floor Plans", floor_img)
        st.metric("📁 Total Images", total)
    
    with col4:
        st.metric("🌟 Good Quality", good_qual, f"{good_qual/total*100:.1f}%")
        st.metric("⚠️ Bad Quality", bad_qual, f"{bad_qual/total*100:.1f}%")
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")
    
    with col6:
        st.metric("💎 Avg Quality", f"{avg_qual:.1f}%")
    
    with col7:
        if enable_text_removal and ocr_reader:
            st.metric("🧹 Avg Text Removed", f"{avg_text_removal:.1f}%")
        else:
            st.metric("🧹 Text Removal", "Disabled")
    
    with col8:
        failed = total - successful
        st.metric("📊 Success Rate", f"{successful/total*100:.1f}%")
    
    # VISUALIZATIONS
    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📈 Visual Analytics</h2>
</div>
""", unsafe_allow_html=True)
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        template = {
            'layout': {
                'paper_bgcolor': 'rgba(30, 15, 50, 0.5)',
                'plot_bgcolor': 'rgba(20, 10, 40, 0.3)',
                'font': {'color': '#e0e7ff'},
                'xaxis': {'gridcolor': 'rgba(139, 92, 246, 0.1)'},
                'yaxis': {'gridcolor': 'rgba(139, 92, 246, 0.1)'}
            }
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['Category', 'Count']
            fig1 = px.bar(
                cat_counts,
                x='Category',
                y='Count',
                title='📊 Category Distribution',
                color='Category',
                color_discrete_sequence=['#8b5cf6', '#7c3aed', '#6d28d9', '#5b21b6']
            )
            fig1.update_layout(template['layout'], height=400)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            qual_counts = df['quality_status'].value_counts().reset_index()
            qual_counts.columns = ['Quality', 'Count']
            fig2 = px.pie(
                qual_counts,
                names='Quality',
                values='Count',
                title='🎯 Quality Distribution',
                color_discrete_sequence=['#10b981', '#ef4444', '#6b7280']
            )
            fig2.update_layout(template['layout'], height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            if len(df_success) > 0:
                qual_by_cat = df_success.groupby('category')['quality_score'].mean().reset_index()
                fig3 = px.bar(
                    qual_by_cat,
                    x='category',
                    y='quality_score',
                    title='📈 Average Quality by Category',
                    color='quality_score',
                    color_continuous_scale='Viridis'
                )
                fig3.update_layout(template['layout'], height=400)
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No successful images to display quality by category")
        
        with col4:
            if len(df_success) > 0 and enable_text_removal and ocr_reader:
                # Text removal distribution
                fig4 = px.histogram(
                    df_success,
                    x='text_removal_pct',
                    nbins=20,
                    title='🧹 Text Removal Distribution',
                    color_discrete_sequence=['#10b981']
                )
                fig4.update_layout(template['layout'], height=400)
                st.plotly_chart(fig4, use_container_width=True)
            elif len(df_success) > 0:
                fig4 = px.histogram(
                    df_success,
                    x='confidence',
                    nbins=20,
                    title='📊 Confidence Distribution',
                    color_discrete_sequence=['#8b5cf6']
                )
                fig4.update_layout(template['layout'], height=400)
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.warning("No successful images to display")
                
    except ImportError:
        st.warning("⚠️ Install plotly: `pip install plotly`")
    
    # DATA TABLE
    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>📋 Detailed Results Table</h2>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cat_filter = st.multiselect(
            "🏷️ Filter by Category",
            options=df['category'].unique().tolist(),
            default=df['category'].unique().tolist()
        )
    
    with col2:
        qual_filter = st.multiselect(
            "🌟 Filter by Quality",
            options=df['quality_status'].unique().tolist(),
            default=df['quality_status'].unique().tolist()
        )
    
    with col3:
        status_filter = st.multiselect(
            "📊 Filter by Status",
            options=df['status'].unique().tolist(),
            default=df['status'].unique().tolist()
        )
    
    filtered_df = df[
        (df['category'].isin(cat_filter)) &
        (df['quality_status'].isin(qual_filter)) &
        (df['status'].isin(status_filter))
    ]
    
    st.info(f"📋 Showing **{len(filtered_df)}** of **{total}** images")
    
    display_columns = ['filename', 'file_size', 'category', 'confidence', 'quality_status', 
                      'quality_score', 'resolution', 'output_size', 'text_removal', 'status']
    
    if 'error' in filtered_df.columns and filtered_df['error'].notna().any():
        display_columns.append('error')
    
    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        height=400
    )
    
    # DOWNLOADS
    st.markdown("""
<div style='background: rgba(30, 15, 50, 0.6); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.3); margin: 2rem 0;'>
    <h2 style='text-align: center; color: #8b5cf6; margin: 0;'>💾 Download Reports & Images</h2>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📄 CSV Report",
            csv,
            f"homes247_{timestamp}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            "📋 JSON Report",
            json_data,
            f"homes247_{timestamp}.json",
            "application/json",
            use_container_width=True
        )
    
    with col3:
        try:
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Results')
                
                summary_data = {
                    'Metric': [
                        'Total Images', 'Successful', 'Failed', 'Gallery Images',
                        'Floor Plans', 'Master Plans', 'Rejected', 'Good Quality',
                        'Bad Quality', 'Avg Confidence', 'Avg Quality', 'Avg Text Removed'
                    ],
                    'Value': [
                        total, successful, total-successful, property_img,
                        floor_img, master_img, rejected_img, good_qual,
                        bad_qual, f"{avg_conf:.2f}%", f"{avg_qual:.2f}%",
                        f"{avg_text_removal:.2f}%" if enable_text_removal and ocr_reader else "Disabled"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            st.download_button(
                "📊 Excel Report",
                output.getvalue(),
                f"homes247_{timestamp}.xlsx",
                "application/vnd.ms-excel",
                use_container_width=True
            )
        except:
            st.info("📦 Install openpyxl: `pip install openpyxl`")
    
    with col4:
        try:
            with st.spinner("📦 Creating ZIP..."):
                zip_buffer = create_download_zip(st.session_state.results, timestamp)
                if zip_buffer:
                    st.download_button(
                        "📦 Complete ZIP",
                        zip_buffer.getvalue(),
                        f"homes247_complete_{timestamp}.zip",
                        "application/zip",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # WELCOME SCREEN
    st.markdown("""
<div style='text-align: center; padding: 3rem; background: rgba(30, 15, 50, 0.6); border-radius: 15px; border: 1px solid rgba(139, 92, 246, 0.3);'>
    <h2 style='color: #8b5cf6; margin: 0 0 2rem 0;'>👋 Welcome to Homes247 AI Dashboard</h2>
    
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; margin-bottom: 2rem;'>
        <div style='padding: 1.5rem; background: rgba(139, 92, 246, 0.1); border-radius: 10px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <h3 style='color: #8b5cf6; margin: 0 0 0.5rem 0;'>AI Classification</h3>
            <p style='color: #c4b5fd; margin: 0;'>Automatic categorization</p>
        </div>
        
        <div style='padding: 1.5rem; background: rgba(139, 92, 246, 0.1); border-radius: 10px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🔍</div>
            <h3 style='color: #8b5cf6; margin: 0 0 0.5rem 0;'>Quality Analysis</h3>
            <p style='color: #c4b5fd; margin: 0;'>Advanced metrics</p>
        </div>
        
        <div style='padding: 1.5rem; background: rgba(139, 92, 246, 0.1); border-radius: 10px;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🏠</div>
            <h3 style='color: #8b5cf6; margin: 0 0 0.5rem 0;'>Auto Watermark</h3>
            <p style='color: #c4b5fd; margin: 0;'>Professional branding</p>
        </div>
    </div>
    
    <div style='background: linear-gradient(135deg, #059669 0%, #047857 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: white; margin: 0 0 1rem 0;'>⚡ COMPLETELY UNLIMITED + AUTO WATERMARK + <span style='color: #10b981;'>AI TEXT REMOVAL ✨</span> ⚡</h3>
        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;'>
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>∞</div>
                <div style='font-weight: bold; color: white;'>Unlimited Images</div>
                <div style='color: #d1fae5; font-size: 0.9rem;'>Upload 1000+ images!</div>
            </div>
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🧹</div>
                <div style='font-weight: bold; color: white;'>AI Text Removal</div>
                <div style='color: #d1fae5; font-size: 0.9rem;'>Remove logos & text!</div>
            </div>
            <div style='background: rgba(255, 255, 255, 0.1); padding: 1rem; border-radius: 8px;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🏠</div>
                <div style='font-weight: bold; color: white;'>Auto Watermark</div>
                <div style='color: #d1fae5; font-size: 0.9rem;'>ALL images branded!</div>
            </div>
        </div>
    </div>
    
    <div style='text-align: left; background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: #8b5cf6; margin: 0 0 1rem 0;'>🚀 Quick Start:</h3>
        <ol style='color: #c4b5fd; margin: 0; padding-left: 1.5rem;'>
            <li>Upload your logo in sidebar (watermark)</li>
            <li>Enable AI Text Removal (optional)</li>
            <li>Upload images above (unlimited!)</li>
            <li>Adjust thresholds (optional)</li>
            <li>Click "Process All Images"</li>
            <li>Download cleaned & watermarked results</li>
        </ol>
    </div>
    
    <div style='text-align: left; background: rgba(139, 92, 246, 0.1); padding: 1.5rem; border-radius: 10px;'>
        <h3 style='color: #8b5cf6; margin: 0 0 1rem 0;'>✨ Features:</h3>
        <div style='color: #c4b5fd; line-height: 1.8;'>
            * ✓ Unlimited images & file sizes<br/>
            * ✓ Persistent statistics tracking<br/>
            * ✓ Upload history with JSON logs<br/>
            * <span style='color: #10b981; font-weight: bold;'>✓ AI-powered text & logo removal ✨ NEW</span><br/>
            * ✓ Automatic watermark on ALL images<br/>
            * ✓ Real-time progress tracking<br/>
            * ✓ Quality analysis<br/>
            * ✓ Auto-resize & organize<br/>
            * ✓ Export CSV, JSON, Excel, ZIP
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# FOOTER
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 15px; margin-top: 3rem;'>
    <h2 style='margin: 0; color: white; font-size: 2rem;'>🏠 HOMES247</h2>
    <p style='margin: 0.5rem 0 0 0; color: #e9d5ff;'>India's Favourite Property Portal</p>
    <p style='margin: 0.5rem 0 0 0; color: #fae8ff; font-weight: bold;'>∞ UNLIMITED EDITION + AUTO WATERMARK + <span style='color: #10b981;'>AI TEXT REMOVAL ✨</span> ∞</p>
    <p style='margin: 1rem 0 0 0; color: #c4b5fd; font-size: 0.9rem;'>Version 2.4 Professional - Zero Limits + Persistent Statistics + Upload History + AI Text Removal + Auto Watermark | Powered by AI</p>
    <p style='margin: 0.5rem 0 0 0; color: #c4b5fd; font-size: 0.8rem;'>© 2026 Homes247. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)