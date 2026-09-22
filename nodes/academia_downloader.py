import os
import asyncio
import requests
import threading
import urllib.parse
import re
import json
import math
import subprocess
import sys
import tempfile
import time
import shutil
from server import PromptServer
from aiohttp import web
import folder_paths

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

ACTIVE_DOWNLOADS = {}
DOWNLOAD_ERRORS = {}


def parse_cli_speed(line):
    """Convert the CLI's byte rate (SI or IEC units) to bytes per second."""
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*([kKMGT]?)(i?)B/s', line)
    if not match:
        return None
    power = {'': 0, 'K': 1, 'M': 2, 'G': 3, 'T': 4}[match.group(2).upper()]
    return float(match.group(1)) * (1024 if match.group(3) else 1000) ** power


def find_hf_cli():
    """Find a working optional CLI without importing or installing dependencies."""
    candidates = []
    executable = shutil.which('hf')
    if executable:
        candidates.append([executable])
    candidates.append([sys.executable, '-c',
                       'import sys; sys.stderr.isatty = lambda: True; '
                       'from huggingface_hub.cli.hf import main; main()'])
    for command in candidates:
        try:
            result = subprocess.run(command + ['download', '--help'],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if result.returncode == 0:
                return command
        except (OSError, subprocess.TimeoutExpired):
            continue
    return None


def parse_hf_file_url(url):
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.strip('/').split('/')
    if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or len(parts) < 5 or parts[2] not in ('resolve', 'blob'):
        raise ValueError('cli requires a Hugging Face file URL (resolve/blob), or a selected repository file.')
    repo_id = '/'.join(parts[:2])
    revision = urllib.parse.unquote(parts[3])
    filename = urllib.parse.unquote('/'.join(parts[4:]))
    if any(p in ('', '.', '..') or ':' in p for p in filename.replace('\\', '/').split('/')):
        raise ValueError('Invalid Hugging Face file path.')
    return repo_id, revision, filename


def background_hf_cli_task(url, file_path, hf_token='', cli_command=None):
    """Run the detected official CLI, forwarding tqdm progress."""
    try:
        repo_id, revision, filename = parse_hf_file_url(url)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        env = os.environ.copy()
        if hf_token:
            env['HF_TOKEN'] = hf_token
        env['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
        env['PYTHONIOENCODING'] = 'utf-8'
        # Same volume for atomic publication; nested repository paths stay private.
        with tempfile.TemporaryDirectory(prefix='.academia-hf-', dir=os.path.dirname(file_path)) as staging:
            command = cli_command + ['download', repo_id, filename, '--revision', revision, '--local-dir', staging]
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  env=env, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)) as process:
                line = ''
                for byte in iter(lambda: process.stdout.read(1), b''):
                    char = byte.decode('utf-8', errors='replace')
                    if char in '\r\n':
                        match = re.search(r'(\d{1,3})%\|', line)
                        if match:
                            ACTIVE_DOWNLOADS[url] = {'progress': min(99, int(match.group(1))), 'method': 'cli',
                                                     'speed_bps': parse_cli_speed(line), 'updated_at': time.monotonic()}
                        line = ''
                    else:
                        line = (line + char)[-4096:]
                if process.wait() != 0:
                    raise RuntimeError('HF CLI failed. Check Hugging Face access/token and connection.')
            source = os.path.join(staging, *filename.split('/'))
            os.replace(source, file_path)
    except Exception as exc:
        DOWNLOAD_ERRORS[url] = str(exc).replace(hf_token, '[token]') if hf_token else str(exc)
    finally:
        ACTIVE_DOWNLOADS.pop(url, None)

# Una sesion HTTP por hilo. Reutiliza la conexion entre las varias peticiones que
# hace cada descarga (comprobar cabeceras, seguir redirecciones, bajar el fichero)
# en vez de abrir una nueva cada vez. Por hilo y no global porque el nodo descarga
# en paralelo y requests.Session no esta pensada para usarse desde varios a la vez.
_LOCAL = threading.local()


def sesion():
    s = getattr(_LOCAL, "http", None)
    if s is None:
        s = requests.Session()
        _LOCAL.http = s
    return s


TOKENS_FILE = os.path.join(folder_paths.base_path, "models", "academia_tokens.json")
PRESETS_DIR = os.path.join(folder_paths.base_path, "models", "academia_presets")
os.makedirs(PRESETS_DIR, exist_ok=True)
URL_INFO_CACHE = {} 

def format_size(size_bytes):
    try:
        size_bytes = int(size_bytes)
        if size_bytes == 0: return "0 B"
        size_name = ("B", "KB", "MB", "GB", "TB")
        i = int(math.floor(math.log(size_bytes, 1024))) if size_bytes > 0 else 0
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_name[i]}"
    except Exception:
        return "Unknown"

# Los parametros de token van vacios por defecto: eso significa "sin token",
# no una credencial escrita en el codigo.
def get_headers_with_auth(url, civitai_token="", hf_token=""):  # nosec B107
    req_headers = HEADERS.copy()
    if "civitai.com" in url and civitai_token:
        req_headers["Authorization"] = f"Bearer {civitai_token}"
    elif "huggingface.co" in url and hf_token:
        req_headers["Authorization"] = f"Bearer {hf_token}"
    return req_headers

# Los parametros de token van vacios por defecto: eso significa "sin token",
# no una credencial escrita en el codigo.
def get_file_info_from_url(url, civitai_token="", hf_token=""):  # nosec B107
    if not url or not url.startswith(('http://', 'https://')):
        return None, "0 B"
        
    if url in URL_INFO_CACHE:
        return URL_INFO_CACHE[url]["filename"], URL_INFO_CACHE[url]["size"]

    try:
        req_headers = get_headers_with_auth(url, civitai_token, hf_token)
        response = sesion().get(url, stream=True, allow_redirects=True, headers=req_headers, timeout=8)
        response.close()
        
        if response.status_code in [401, 403] or "civitai.com/login" in response.url:
            return None, "Auth Required"

        size_bytes = response.headers.get('Content-Length')
        formatted_size = format_size(size_bytes) if size_bytes else "Unknown"

        fname = None
        cd = response.headers.get('Content-Disposition')
        if cd:
            match = re.search(r'filename=["\']?([^;"\']+)', cd)
            if match: fname = os.path.basename(match.group(1).strip())
            
        if not fname:
            parsed = urllib.parse.urlparse(response.url)
            fname = os.path.basename(parsed.path)
            if not fname or fname.isdigit():
                fname = (fname if fname else "model") + ".safetensors"

        URL_INFO_CACHE[url] = {"filename": fname, "size": formatted_size}
        return fname, formatted_size
    except Exception as e:
        return None, "Unknown"

def _dentro_de(base, destino):
    """True si destino queda dentro de base. Distinta unidad cuenta como fuera."""
    base, destino = os.path.abspath(base), os.path.abspath(destino)
    try:
        return os.path.commonpath([base, destino]) == base
    except ValueError:
        return False


def _sub_seguro(base, subfolder):
    """Une subfolder a base sin permitir que se salga.

    Quitar ".." no basta: en Windows una ruta con unidad ("C:\\Windows\\Temp")
    hace que os.path.join descarte la base por completo. Devuelve None si el
    valor recibido intenta escapar.
    """
    if not subfolder:
        return base
    limpio = subfolder.replace("..", "").strip("\\/")
    if not limpio:
        return base
    destino = os.path.join(base, limpio)
    return destino if _dentro_de(base, destino) else None


def find_existing_file(folder_name, subfolder, filename):
    paths = folder_paths.get_folder_paths(folder_name)
    if not paths: return None
    for base_path in paths:
        check_path = _sub_seguro(base_path, subfolder)
        if check_path is None:
            continue
        full_file_path = os.path.join(check_path, os.path.basename(filename or ""))
        if os.path.exists(full_file_path):
            return full_file_path
    return None

def get_download_target_path(folder_name, subfolder):
    paths = folder_paths.get_folder_paths(folder_name)
    target_base = paths[0] if paths else os.path.join(folder_paths.base_path, "models", folder_name)
    for p in paths or []:
        if os.path.basename(os.path.normpath(p)) == folder_name:
            target_base = p
            break
    return _sub_seguro(target_base, subfolder)

# Los parametros de token van vacios por defecto: eso significa "sin token",
# no una credencial escrita en el codigo.
def background_download_task(url, file_path, civitai_token="", hf_token=""):  # nosec B107
    temp_path = file_path + ".temp"
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        req_headers = get_headers_with_auth(url, civitai_token, hf_token)
        # timeout de conexion y de lectura entre trozos: sin el, un servidor que
        # deja la conexion abierta sin enviar nada cuelga la descarga para siempre.
        with sesion().get(url, stream=True, allow_redirects=True, headers=req_headers,
                          timeout=(10, 60)) as r:
            r.raise_for_status()
            total_length = r.headers.get('content-length')
            total_length = int(total_length) if total_length else 0
            downloaded = 0
            sample_time, sample_bytes, speed = time.monotonic(), 0, 0.0
            
            with open(temp_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): 
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - sample_time >= 0.5:
                            speed = (downloaded - sample_bytes) / (now - sample_time)
                            sample_time, sample_bytes = now, downloaded
                        ACTIVE_DOWNLOADS[url] = {
                            "progress": min(99, int(downloaded / total_length * 100)) if total_length else -1,
                            "speed_bps": speed, "updated_at": now,
                        }
        
        os.replace(temp_path, file_path)
    except Exception as e:
        DOWNLOAD_ERRORS[url] = 'Download failed. Check the URL, access token and connection.'
        if os.path.exists(temp_path): os.remove(temp_path)
    finally:
        ACTIVE_DOWNLOADS.pop(url, None)

# --- RUTAS API ---
# Placeholder sent to the browser instead of a stored token. Posting it back
# means "keep whatever is already saved", so the UI can round-trip without ever
# handling the real value.
TOKEN_MASK = "****"  # nosec B105


def _read_tokens():
    """Tokens guardados, o {} si el fichero falta o esta corrupto."""
    try:
        with open(TOKENS_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _resolve_token(value, provider):
    return _read_tokens().get(provider, '') if value == TOKEN_MASK else value


@PromptServer.instance.routes.get("/academia/tokens")
async def get_tokens(request):
    # Never return the stored tokens: this endpoint is reachable by anything
    # that can talk to the ComfyUI port. Report only whether one is set.
    saved = _read_tokens()
    return web.json_response({
        k: (TOKEN_MASK if saved.get(k) else "")
        for k in ("civitai", "huggingface")
    })


@PromptServer.instance.routes.post("/academia/tokens")
async def save_tokens(request):
    data = await request.json()
    saved = _read_tokens()
    try:
        for k in ("civitai", "huggingface"):
            value = data.get(k, "")
            # The UI echoes the mask back when the field was not edited.
            if value != TOKEN_MASK:
                saved[k] = value
        with open(TOKENS_FILE, "w") as f:
            f.write(json.dumps(saved))
        return web.json_response({"status": "success"})
    except Exception:
        return web.json_response({"status": "error", "message": "Could not save tokens"}, status=400)

def _preset_path(name):
    """Ruta de <name>.json dentro de PRESETS_DIR, o None si se sale del directorio."""
    if not name:
        return None
    base = os.path.abspath(PRESETS_DIR)
    destino = os.path.abspath(os.path.join(base, f"{name}.json"))
    if os.path.commonpath([base, destino]) != base:
        return None
    return destino


@PromptServer.instance.routes.get("/academia/download_presets")
async def get_download_presets(request):
    name = request.query.get("name")
    if name:
        filepath = _preset_path(name)
        if filepath is None:
            return web.json_response({"status": "error", "message": "Invalid name"}, status=400)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f: return web.json_response({"status": "success", "data": json.load(f)})
        return web.json_response({"status": "error", "message": "Preset not found"})
    files = [f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json")]
    return web.json_response({"status": "success", "files": files})

@PromptServer.instance.routes.post("/academia/download_presets")
async def save_download_preset(request):
    data = await request.json()
    name = data.get("name")
    if not name: return web.json_response({"status": "error", "message": "No name provided"})
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    try:
        with open(os.path.join(PRESETS_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(data.get("data", []), indent=4))
        return web.json_response({"status": "success"})
    except Exception: return web.json_response({"status": "error", "message": "Could not save the preset"}, status=400)

@PromptServer.instance.routes.get("/academia/folders")
async def get_folders(request):
    raw_folders = list(folder_paths.folder_names_and_paths.keys())
    for ef in ["checkpoints", "unet", "diffusion_models", "loras", "vae", "text_encoders", "clip", "controlnet", "upscale_models", "embeddings"]:
        if ef not in raw_folders: raw_folders.append(ef)
    raw_folders.sort()
    return web.json_response(raw_folders)

@PromptServer.instance.routes.post("/academia/parse_url")
async def parse_url(request):
    data = await request.json()
    url = data.get("url", "")
    hf_token = _resolve_token(data.get("hf_token", ""), 'huggingface')
    if "huggingface.co" in url and "/resolve/" not in url and "/blob/" not in url:
        match = re.search(r"huggingface\.co/([^/]+/[^/?#]+)(?:/tree/([^/?#]+))?", url)
        if match:
            repo_id, branch = match.group(1), match.group(2) or "main"
            headers = HEADERS.copy()
            if hf_token: headers["Authorization"] = f"Bearer {hf_token}"
            try:
                res = await asyncio.to_thread(sesion().get, f"https://huggingface.co/api/models/{repo_id}", headers=headers, timeout=10)
                if res.status_code == 200:
                    files = [{"name": os.path.basename(s["rfilename"]), "url": f"https://huggingface.co/{repo_id}/resolve/{branch}/{s['rfilename']}", "size": format_size(s.get("size"))} 
                             for s in res.json().get("siblings", []) if s["rfilename"].endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".bin", ".pth", ".onnx", ".sft"))]
                    if files: return web.json_response({"status": "success", "type": "repo", "files": files})
            except Exception: pass
    return web.json_response({"status": "success", "type": "direct", "url": url})

@PromptServer.instance.routes.post("/academia/check")
async def check_file(request):
    data = await request.json()
    url = data.get("url", "").strip() 
    folder = data.get("folder")
    subfolder = data.get("subfolder", "").strip()
    filename_hint = data.get("filename", "").strip() 
    civ_t, hf_t = data.get("civitai_token", "").strip(), data.get("hf_token", "").strip()
    
    if not url or url == "none": return web.json_response({"status": "error", "exists": False})

    if url in ACTIVE_DOWNLOADS:
        state = ACTIVE_DOWNLOADS.get(url, {})
        speed = state.get("speed_bps")
        if time.monotonic() - state.get("updated_at", 0) > 3:
            speed = 0
        return web.json_response({"status": "success", "exists": False, "is_downloading": True,
                                  "progress": state.get("progress", -1), "speed_bps": speed})

    if url in DOWNLOAD_ERRORS:
        return web.json_response({"status": "error", "exists": False, "is_downloading": False, "message": DOWNLOAD_ERRORS[url]})

    filename = os.path.basename(filename_hint) if filename_hint else ""
    filesize = "Unknown"

    if not filename or filename in ["Direct Link", "Pending..."]:
        filename, filesize = await asyncio.to_thread(get_file_info_from_url, url, civ_t, hf_t)
        if not filename: return web.json_response({"status": "error", "exists": False, "message": "auth_required", "filesize": filesize})

    existing_file = find_existing_file(folder, subfolder, filename)
    exists = existing_file is not None

    if exists:
        try: filesize = format_size(os.path.getsize(existing_file))
        except Exception: pass
        URL_INFO_CACHE[url] = {"filename": filename, "size": filesize}
    else:
        # ¡BUGFIX! Forzamos siempre a leer el tamaño si no existe, sin importar si el nombre ya lo sabíamos.
        real_fname, real_fsize = await asyncio.to_thread(get_file_info_from_url, url, civ_t, hf_t)
        if real_fsize and real_fsize != "Unknown":
            filesize = real_fsize
            
        if real_fname and real_fname != filename:
            filename = os.path.basename(real_fname)
            exists = find_existing_file(folder, subfolder, filename) is not None
            if exists:
                try: filesize = format_size(os.path.getsize(find_existing_file(folder, subfolder, filename)))
                except Exception: pass

    return web.json_response({"status": "success", "exists": exists, "filename": filename, "filesize": filesize, "is_downloading": False})

@PromptServer.instance.routes.post("/academia/download")
async def download_file(request):
    data = await request.json()
    url = data.get("url", "").strip()
    folder, subfolder = data.get("folder"), data.get("subfolder", "").strip()
    filename = data.get("filename", "").strip()
    civ_t, hf_t = data.get("civitai_token", "").strip(), data.get("hf_token", "").strip()
    civ_t, hf_t = _resolve_token(civ_t, 'civitai'), _resolve_token(hf_t, 'huggingface')
    method = data.get('method', 'http')
    if method not in ('http', 'cli'):
        return web.json_response({'status': 'error', 'message': 'Invalid download method.'}, status=400)
    if method == 'cli':
        cli_command = await asyncio.to_thread(find_hf_cli)
        if cli_command is None:
            return web.json_response({'status': 'error', 'code': 'hf_cli_missing'}, status=400)
        try:
            _, _, hf_filename = parse_hf_file_url(url)
        except ValueError as exc:
            return web.json_response({'status': 'error', 'message': str(exc)}, status=400)
        if not filename or filename in ('Direct Link', 'Pending...'):
            filename = hf_filename.split('/')[-1]
    
    if not url: return web.json_response({"status": "error", "message": "Invalid URL."})
    if url in ACTIVE_DOWNLOADS: return web.json_response({"status": "started", "message": "Already downloading."})

    filename = os.path.basename(filename) if filename else ""
    if not filename or filename in ["Direct Link", "Pending..."]:
        filename, _ = await asyncio.to_thread(get_file_info_from_url, url, civ_t, hf_t)
        if not filename: return web.json_response({"status": "error", "message": "Auth required or invalid link."})

    if find_existing_file(folder, subfolder, filename):
        return web.json_response({"status": "exists", "message": "File already exists."})

    destino = get_download_target_path(folder, subfolder)
    if destino is None:
        return web.json_response({"status": "error", "message": "Invalid destination folder."}, status=400)

    file_path = os.path.join(destino, filename)
    DOWNLOAD_ERRORS.pop(url, None)
    ACTIVE_DOWNLOADS[url] = {"progress": 0}
    if method == 'cli':
        asyncio.create_task(asyncio.to_thread(background_hf_cli_task, url, file_path, hf_t, cli_command))
    else:
        asyncio.create_task(asyncio.to_thread(background_download_task, url, file_path, civ_t, hf_t))
    
    return web.json_response({"status": "started"})

class AcademiaDownloaderNode:
    def __init__(self): pass
    @classmethod
    def INPUT_TYPES(s): return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "Academia SD"
    OUTPUT_NODE = True
    def do_nothing(self): return ()

NODE_CLASS_MAPPINGS = {"AcademiaSD_Downloader": AcademiaDownloaderNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AcademiaSD_Downloader": "Academia SD Automatic downloader ⬇️"}
