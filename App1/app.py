# Author: vlarobbyk (Modified Layout + Auto-resize)
# Version: 2.1
# Date: 2025-11-16

from flask import Flask, render_template, Response
import time
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

app = Flask(__name__)

_URL = 'http://192.168.0.104'
_PORT = '81'
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])
prev_time = time.time()
fps = 0.0

REFERENCE_WIDTH = 320
REFERENCE_HEIGHT = 240

def video_capture():
    global prev_time, fps, REFERENCE_WIDTH, REFERENCE_HEIGHT
    
    try:
        from flask import request
    except Exception:
        request = None

    gauss_mean = float(request.args.get('gauss_mean', 0.0)) if request else 0.0
    gauss_sigma = float(request.args.get('gauss_sigma', 20.0)) if request else 20.0
    gauss_prob = float(request.args.get('gauss_prob', 0.0)) if request else 0.0
    speck_var = float(request.args.get('speck_var', 0.04)) if request else 0.04
    speck_prob = float(request.args.get('speck_prob', 0.0)) if request else 0.0
    pytorch_kernel = int(request.args.get('pt_k', 5)) if request else 5
    pytorch_sigma = float(request.args.get('pt_sigma', 1.2)) if request else 1.2

    bg_median_gray = calcular_fondo()
    print(f"Fondo cargado: {bg_median_gray.shape}")

    res = requests.get(stream_url, stream=True)
    bytes_buffer = b''

    median_mask = torch.ones(3, 3, dtype=torch.float32) / 9.0
    kernel = median_mask.unsqueeze(0).unsqueeze(0)

    for chunk in res.iter_content(chunk_size=1024):
        bytes_buffer += chunk
        a = bytes_buffer.find(b'\xff\xd8')
        b = bytes_buffer.find(b'\xff\xd9')
        
        if a != -1 and b != -1 and b > a:
            jpg = bytes_buffer[a:b+2]
            bytes_buffer = bytes_buffer[b+2:]

            cv_img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if cv_img is None:
                continue

            try:
                current_h, current_w = cv_img.shape[:2]
                
                if REFERENCE_WIDTH is None or REFERENCE_HEIGHT is None:
                    REFERENCE_WIDTH = current_w
                    REFERENCE_HEIGHT = current_h
                    print(f"Dimensiones de referencia establecidas: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
                
                if current_w != REFERENCE_WIDTH or current_h != REFERENCE_HEIGHT:
                    cv_img = cv2.resize(cv_img, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                    print(f"Frame redimensionado de {current_w}x{current_h} a {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
                
                h, w = REFERENCE_HEIGHT, REFERENCE_WIDTH
                
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) != 0 else fps
                prev_time = current_time
                fps_text = f"FPS: {fps:.2f}"

                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # 1) Ruido
                N = 537
                noise = np.zeros((h, w), dtype=np.uint8)
                noise[np.random.randint(0,h,N), np.random.randint(0,w,N)] = 255
                noise_image = cv2.bitwise_or(gray, noise)

                # 2) Convolución
                img_tensor = torch.from_numpy(noise_image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                convolved_output = F.conv2d(img_tensor, kernel, padding=1)
                img_output = cv2.convertScaleAbs(convolved_output.cpu().numpy().squeeze())

                # 3) Diferencia con fondo (redimensionar fondo si es necesario)
                if bg_median_gray.shape != (h, w):
                    bg_resized = cv2.resize(bg_median_gray, (w, h))
                else:
                    bg_resized = bg_median_gray
                    
                diff = cv2.absdiff(gray, bg_resized)
                _, fondo_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                km = np.ones((3,3), np.uint8)
                fondo_mask = cv2.morphologyEx(fondo_mask, cv2.MORPH_OPEN, km)

                # 4) Mejoras de iluminación
                hist_eq, clahe, retinex = mejorar_iluminacion(gray)

                # 5) Extraer objeto
                solo_objeto = extraer_objeto(cv_img, fondo_mask)

                # 6) Ruido en color
                noisy_color = cv_img.copy()
                if gauss_prob > 0:
                    noisy_color = add_gaussian_noise_color(noisy_color, mean=gauss_mean, sigma=gauss_sigma, prob=gauss_prob)
                if speck_prob > 0:
                    noisy_color = add_speckle_noise_color(noisy_color, var=speck_var, prob=speck_prob)

                # 7) PyTorch denoise
                try:
                    denoised_torch = pytorch_gaussian_conv_denoise(noisy_color, kernel_size=pytorch_kernel, sigma=pytorch_sigma, device='cpu')
                except Exception as e:
                    print("Warning: pytorch denoise failed:", e)
                    denoised_torch = noisy_color.copy()

                # 8) Comparación
                diff_mask = cv2.cvtColor(cv2.absdiff(cv_img, denoised_torch), cv2.COLOR_BGR2GRAY)
                _, diff_mask_bin = cv2.threshold(diff_mask, 20, 255, cv2.THRESH_BINARY)
                comparison_img = copy_to_comparison_pythonic(cv_img, denoised_torch, mask=diff_mask_bin)

                # 9) Filtros espaciales
                spatial_results = apply_spatial_filters(noisy_color, kernel_sizes=(3,5,7))

                # 10) Bordes
                gray_noisy = cv2.cvtColor(noisy_color, cv2.COLOR_BGR2GRAY)
                edges_no_smooth = compute_edges(gray_noisy, use_smoothing=None)
                edges_median3 = compute_edges(gray_noisy, use_smoothing=('median', 3))
                edges_gauss5 = compute_edges(gray_noisy, use_smoothing=('gaussian', 5))

                COLS = 4
                TITLE_HEIGHT = 30
                
                images_to_show = []
                
                # FILA 1: Procesamiento básico
                images_to_show.append(("ORIGINAL", cv_img))
                images_to_show.append(("GRISES", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("RUIDO", cv2.cvtColor(noise_image, cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("CONVOLUCION", cv2.cvtColor(img_output, cv2.COLOR_GRAY2BGR)))
                
                # FILA 2: Fondo y objeto
                images_to_show.append(("MASCARA FONDO", cv2.cvtColor(fondo_mask, cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("OBJETO", solo_objeto))
                images_to_show.append(("COLOR RUIDOSO", noisy_color))
                images_to_show.append(("PT SUAVIZADO", denoised_torch))
                
                # FILA 3: Mejoras de iluminación
                images_to_show.append(("ECUALIZACION", cv2.cvtColor(hist_eq, cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("CLAHE", cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("RETINEX", cv2.cvtColor(retinex, cv2.COLOR_GRAY2BGR)))
                if comparison_img.shape[1] >= 2*w:
                    images_to_show.append(("COMP ORIG", comparison_img[:, 0:w]))
                else:
                    images_to_show.append(("COMP ORIG", cv_img))
                
                # FILA 4: Comparación y filtros
                if comparison_img.shape[1] >= 2*w:
                    images_to_show.append(("COMP SUAVIZADO", comparison_img[:, w:2*w]))
                else:
                    images_to_show.append(("COMP SUAVIZADO", denoised_torch))
                images_to_show.append(("MEDIANA 3", spatial_results.get("median_3", noisy_color)))
                images_to_show.append(("DIFUMINADO 3", spatial_results.get("blur_3", noisy_color)))
                images_to_show.append(("GAUSSIANO 3", spatial_results.get("gaussian_3", noisy_color)))
                
                # FILA 5: Más filtros espaciales
                images_to_show.append(("MEDIANA 5", spatial_results.get("median_5", noisy_color)))
                images_to_show.append(("DIFUMINADO 5", spatial_results.get("blur_5", noisy_color)))
                images_to_show.append(("GAUSSIANO 5", spatial_results.get("gaussian_5", noisy_color)))
                images_to_show.append(("MEDIANA 7", spatial_results.get("median_7", noisy_color)))
                
                # FILA 6: Últimos filtros
                images_to_show.append(("DIFUMINADO 7", spatial_results.get("blur_7", noisy_color)))
                images_to_show.append(("GAUSSIANO 7", spatial_results.get("gaussian_7", noisy_color)))
                images_to_show.append(("CANNY CRUDO", cv2.cvtColor(edges_no_smooth['canny'], cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("SOBEL CRUDO", cv2.cvtColor(edges_no_smooth['sobel'], cv2.COLOR_GRAY2BGR)))
                
                # FILA 7: Bordes con filtros
                images_to_show.append(("CANNY MED3", cv2.cvtColor(edges_median3['canny'], cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("SOBEL MED3", cv2.cvtColor(edges_median3['sobel'], cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("CANNY GAUSS5", cv2.cvtColor(edges_gauss5['canny'], cv2.COLOR_GRAY2BGR)))
                images_to_show.append(("SOBEL GAUSS5", cv2.cvtColor(edges_gauss5['sobel'], cv2.COLOR_GRAY2BGR)))
                
                total_images = len(images_to_show)
                rows = (total_images + COLS - 1) // COLS
                
                total_height = rows * (h + TITLE_HEIGHT)
                total_width = COLS * w
                canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)
                
                for idx, (title, img) in enumerate(images_to_show):
                    row = idx // COLS
                    col = idx % COLS
                    
                    if img.shape[0] != h or img.shape[1] != w:
                        img = cv2.resize(img, (w, h))
                    
                    y_start = row * (h + TITLE_HEIGHT) + TITLE_HEIGHT
                    y_end = y_start + h
                    x_start = col * w
                    x_end = x_start + w
                    
                    canvas[y_start:y_end, x_start:x_end] = img
                    
                    title_y = row * (h + TITLE_HEIGHT) + 22
                    cv2.putText(canvas, title, (x_start + 10, title_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                cv2.putText(canvas, fps_text, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                ok, encoded = cv2.imencode(".jpg", canvas)
                if ok:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" +
                        encoded.tobytes() +
                        b"\r\n"
                    )

            except Exception as e:
                print("ERROR:", e)
                import traceback
                traceback.print_exc()
                continue

# FUNCIONES
def extraer_objeto(frame_bgr, mask):
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    objeto = cv2.bitwise_and(frame_bgr, mask_3ch)
    return objeto

def mejorar_iluminacion(gray):
    hist_eq = cv2.equalizeHist(gray)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe_obj.apply(gray)
    img_f = gray.astype(np.float32) + 1.0
    log_img = np.log(img_f)
    log_blur = cv2.GaussianBlur(img_f, (15,15), 30)
    log_blur = np.log(log_blur + 1.0)
    retinex = log_img - log_blur
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return hist_eq, clahe, retinex

def calcular_fondo():
    global REFERENCE_WIDTH, REFERENCE_HEIGHT
    print("Calculando fondo... espera 2 segundos...")
    res = requests.get(stream_url, stream=True)
    bytes_buffer = b''
    bg_frames = []
    
    while len(bg_frames) < 30:
        for chunk in res.iter_content(chunk_size=1024):
            bytes_buffer += chunk
            a = bytes_buffer.find(b'\xff\xd8')
            b = bytes_buffer.find(b'\xff\xd9')
            if a != -1 and b != -1 and b > a:
                jpg = bytes_buffer[a:b+2]
                bytes_buffer = bytes_buffer[b+2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    if REFERENCE_WIDTH is None or REFERENCE_HEIGHT is None:
                        REFERENCE_HEIGHT, REFERENCE_WIDTH = frame.shape[:2]
                        print(f"Dimensiones detectadas del stream: {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}")
                    
                    if frame.shape[1] != REFERENCE_WIDTH or frame.shape[0] != REFERENCE_HEIGHT:
                        frame = cv2.resize(frame, (REFERENCE_WIDTH, REFERENCE_HEIGHT))
                    
                    bg_frames.append(frame)
                    print(f"Frames capturados: {len(bg_frames)}/30")
                break
    
    bg = np.median(np.array(bg_frames), axis=0).astype(np.uint8)
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

def add_gaussian_noise_color(img_bgr, mean=0.0, sigma=20.0, prob=1.0):
    h, w, c = img_bgr.shape
    gauss = np.random.normal(mean, sigma, (h, w, c)).astype(np.float32)
    mask = np.random.rand(h, w, 1) < prob
    noisy = img_bgr.astype(np.float32) + gauss * mask.astype(np.float32)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def add_speckle_noise_color(img_bgr, var=0.04, prob=1.0):
    h, w, c = img_bgr.shape
    gauss = np.random.normal(0, np.sqrt(var), (h, w, c)).astype(np.float32)
    mask = np.random.rand(h, w, 1) < prob
    img_f = img_bgr.astype(np.float32)
    noisy = img_f + img_f * (gauss * mask.astype(np.float32))
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def apply_spatial_filters(img_bgr, kernel_sizes=(3,5,7)):
    results = {}
    for k in kernel_sizes:
        k_odd = k if k % 2 == 1 else k+1
        results[f"median_{k_odd}"] = cv2.medianBlur(img_bgr, k_odd)
        results[f"blur_{k_odd}"] = cv2.blur(img_bgr, (k_odd, k_odd))
        results[f"gaussian_{k_odd}"] = cv2.GaussianBlur(img_bgr, (k_odd, k_odd), 0)
    return results

def pytorch_gaussian_conv_denoise(img_bgr, kernel_size=5, sigma=1.5, device='cpu'):
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    kernel_t = torch.from_numpy(kernel.astype(np.float32))
    c = 3
    weight = kernel_t.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1).to(device)
    img_f = img_bgr.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_f.transpose(2, 0, 1)).unsqueeze(0).to(device)
    pad = kernel_size // 2
    img_t = F.pad(img_t, (pad, pad, pad, pad), mode='reflect')
    denoised_t = F.conv2d(img_t, weight, bias=None, stride=1, padding=0, groups=c)
    denoised = denoised_t.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    denoised = np.clip(denoised * 255.0, 0, 255).astype(np.uint8)
    return denoised

def copy_to_comparison_pythonic(orig, denoised, mask=None):
    h, w, _ = orig.shape
    left = orig.copy()
    if mask is None:
        diff = cv2.absdiff(orig, denoised)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 20, 255, cv2.THRESH_BINARY)
    else:
        mask = mask.copy()
    mask_bool = mask.astype(bool)
    dst = orig.copy()
    for ch in range(3):
        ch_dst = dst[:, :, ch]
        ch_src = denoised[:, :, ch]
        ch_dst[mask_bool] = ch_src[mask_bool]
        dst[:, :, ch] = ch_dst
    out = np.zeros((h, w*2, 3), dtype=np.uint8)
    out[:, 0:w] = left
    out[:, w:w*2] = dst
    return out

def compute_edges(gray, use_smoothing=None):
    proc = gray
    if use_smoothing is not None:
        typ, k = use_smoothing
        k_odd = k if k % 2 == 1 else k+1
        if typ == 'median':
            proc = cv2.medianBlur(gray, k_odd)
        elif typ == 'gaussian':
            proc = cv2.GaussianBlur(gray, (k_odd, k_odd), 0)
    canny = cv2.Canny(proc, 50, 150)
    sobelx = cv2.Sobel(proc, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(proc, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.clip(sobel / sobel.max() * 255.0, 0, 255).astype(np.uint8)
    return {'canny': canny, 'sobel': sobel}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False, port=5005)
