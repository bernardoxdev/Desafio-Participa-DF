import os

from flask import Flask, render_template, send_from_directory, abort, request
from werkzeug.utils import secure_filename

from libs.sheets import DataAPI

BASE_DIR = os.path.dirname(__file__)
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, "data", "processada")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "raw")
TRAIN_CLAS_FOLDER = os.path.join(BASE_DIR, "data", "train", "clas")
TRAIN_CONTEXT_FOLDER = os.path.join(BASE_DIR, "data", "train", "context")
ALLOWED_EXTENSIONS = {"pdf", "csv", "txt", "xlsx"}

dataAPI = DataAPI()

app = Flask(__name__, template_folder="../frontend/templates")

app.static_folder = '../frontend/src'
app.static_url_path = '/static'

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret")
app.config['SESSION_TYPE'] = 'filesystem'
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["TRAIN_CLAS_FOLDER"] = TRAIN_CLAS_FOLDER
app.config["TRAIN_CONTEXT_FOLDER"] = TRAIN_CONTEXT_FOLDER

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False
)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    arquivos = os.listdir(DOWNLOAD_FOLDER)
    arquivos_clas = os.listdir(TRAIN_CLAS_FOLDER)
    arquivo_context = os.listdir(TRAIN_CONTEXT_FOLDER)
    
    return render_template('index.html', arquivos=arquivos, arquivos_contexto=arquivo_context, arquivos_classificacao=arquivos_clas)

@app.route("/download/<path:nome_arquivo>", methods=['GET'])
def download(nome_arquivo):
    try:
        return send_from_directory(
            DOWNLOAD_FOLDER,
            nome_arquivo,
            as_attachment=True
        )
    except FileNotFoundError:
        abort(404)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"error": "Arquivo não enviado"}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "Arquivo inválido"}, 400

    if not allowed_file(file.filename):
        return {"error": "Tipo de arquivo não permitido"}, 403

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    nome_documento = request.form.get("nome_documento")
    formato_saida = True if request.form.get("formato_saida") == 'xlsx' else False
    arquivos_contexto = request.form.getlist("arquivos_contexto")
    arquivos_classificacao = request.form.getlist("arquivos_classificacao")
    
    return {
        "message": "Upload realizado com sucesso",
        "file": filename,
        "nome_documento": nome_documento,
        "formato_saida": formato_saida,
        "contexto": arquivos_contexto,
        "classificacao": arquivos_classificacao
    }, 200

@app.route("/uploadContext", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"error": "Arquivo não enviado"}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "Arquivo inválido"}, 400

    if not allowed_file(file.filename):
        return {"error": "Tipo de arquivo não permitido"}, 403

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["TRAIN_CONTEXT_FOLDER"], filename))

    return {"message": "Upload realizado com sucesso"}, 200

@app.route("/uploadClas", methods=["POST"])
def upload():
    if "file" not in request.files:
        return {"error": "Arquivo não enviado"}, 400

    file = request.files["file"]

    if file.filename == "":
        return {"error": "Arquivo inválido"}, 400

    if not allowed_file(file.filename):
        return {"error": "Tipo de arquivo não permitido"}, 403

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["TRAIN_CLAS_FOLDER"], filename))

    return {"message": "Upload realizado com sucesso"}, 200

if __name__ == "__main__":
    # app = DataAPI()
    
    # app.read_docs()
    
    # df = app.get_data_frame(1)
    
    # app.clear_text_col(df, 2)
    
    # print(df.head())
    
    # app.make_analise(df)
    app.run(port="5000", debug=True)