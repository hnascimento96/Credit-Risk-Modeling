## Instalação

```bash
git clone <url-do-repo>
cd Credit-Risk-Modeling
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Usuários de macOS:** evite clonar o projeto em pastas sincronizadas com o
> iCloud (como `~/Documents` ou `~/Desktop`). O iCloud pode marcar arquivos
> como ocultos, e o Python 3.13+ ignora o arquivo `.pth` que registra o
> pacote, causando `ModuleNotFoundError: No module named 'src'`.
>
> Se acontecer, rode:
> ```bash
> chflags nohidden .venv/lib/python3.*/site-packages/__editable__*.pth
> ```
> e reinicie o kernel do notebook.