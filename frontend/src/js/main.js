const tooltipTriggerList = [].slice.call(
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
);
tooltipTriggerList.map(el => new bootstrap.Tooltip(el));

const checkbox = document.getElementById("modoAvancado");
const fieldset = document.getElementById("opcoesAvancadas");

checkbox.addEventListener("change", () => {
    fieldset.disabled = !checkbox.checked;
});

function atualizarContador(selectId, contadorId) {
    const select = document.getElementById(selectId);
    const contador = document.getElementById(contadorId);

    if (!select || !contador) return;

    select.addEventListener("change", () => {
        const total = select.selectedOptions.length;
        contador.innerText = `${total} selecionado${total !== 1 ? "s" : ""}`;
    });
}

atualizarContador("arquivosContexto", "contadorContexto");
atualizarContador("arquivosClassificacao", "contadorClassificacao");

const form = document.getElementById("formDadosAnalise");
const status = document.getElementById("status");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const formData = new FormData(form);

    if (!formData.get("file") || formData.get("file").size === 0) {
        status.innerText = "Selecione um arquivo para upload.";
        return;
    }

    status.innerText = "Enviando arquivo... ⏳";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        let data = {};
        const contentType = response.headers.get("content-type");

        if (contentType && contentType.includes("application/json")) {
            data = await response.json();
        }

        if (response.ok) {
            status.innerText = "Upload realizado com sucesso ✅";
        } else {
            status.innerText = data.error || `Erro ${response.status} ❌`;
        }

    } catch (err) {
        console.error(err);
        status.innerText = "Erro de conexão ❌";
    }
});