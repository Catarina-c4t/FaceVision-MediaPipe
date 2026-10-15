#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceScan AI - Servidor Local em Python
Inicia um servidor web local com suporte a API REST para salvar e listar rostos cadastrados.
"""

import http.server
import socketserver
import json
import os
import sys
import webbrowser
import urllib.parse

# Garante compatibilidade de saída UTF-8 no console do Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PORT = 8000
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces_db.json")

def load_faces():
    """Carrega os dados dos rostos salvos do arquivo JSON."""
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Aviso] Falha ao ler {DATA_FILE}: {e}")
        return []

def save_faces(faces):
    """Salva a lista de rostos no arquivo JSON."""
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(faces, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[Erro] Falha ao salvar em {DATA_FILE}: {e}")
        return False

class FaceAppHandler(http.server.SimpleHTTPRequestHandler):
    """Manipulador HTTP com suporte a rotas estáticas e API /api/faces."""

    def end_headers(self):
        # Permite requisições sem problemas de cache durante o desenvolvimento
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # Endpoint da API para listar rostos
        if parsed.path == "/api/faces":
            faces = load_faces()
            data = json.dumps(faces).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        # Rota padrão: se acessar '/', entrega o index.html
        if parsed.path == "/" or parsed.path == "":
            self.path = "/index.html"

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        # Endpoint da API para cadastrar um novo rosto
        if parsed.path == "/api/faces":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                new_face = json.loads(body)
                faces = load_faces()

                # Verifica se já existe um id correspondente para atualizar ou adiciona
                existing_idx = next((i for i, f in enumerate(faces) if f.get("id") == new_face.get("id")), None)
                if existing_idx is not None:
                    faces[existing_idx] = new_face
                else:
                    faces.append(new_face)

                save_faces(faces)

                response_data = json.dumps({"status": "success", "count": len(faces)}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
            except Exception as e:
                err_data = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_data)))
                self.end_headers()
                self.wfile.write(err_data)
            return

        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)

        # Endpoint para deletar rosto por ID: /api/faces?id=...
        if parsed.path == "/api/faces":
            query = urllib.parse.parse_qs(parsed.query)
            face_id = query.get("id", [None])[0]

            faces = load_faces()
            initial_count = len(faces)
            if face_id:
                faces = [f for f in faces if f.get("id") != face_id]
            else:
                # Se não passar id, limpa todos
                faces = []

            save_faces(faces)

            response_data = json.dumps({
                "status": "success",
                "deleted": initial_count - len(faces),
                "remaining": len(faces)
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_data)))
            self.end_headers()
            self.wfile.write(response_data)
            return

        self.send_response(404)
        self.end_headers()

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), FaceAppHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print("=" * 60)
        print(f"🚀 Servidor do FaceScan AI iniciado!")
        print(f"📍 Acesse em seu navegador: {url}")
        print("💡 Para encerrar, aperte Ctrl+C no terminal.")
        print("=" * 60)
        
        # Abre o navegador automaticamente
        try:
            webbrowser.open(url)
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Servidor encerrado com sucesso.")

if __name__ == "__main__":
    run_server()
