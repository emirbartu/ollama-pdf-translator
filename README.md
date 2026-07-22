<a id="readme-top"></a>

<div align="center">
  <a href="https://github.com/emirbartu/ollama-pdf-translator"></a>

  <h3 align="center">Ollama PDF Translator</h3>

  <p align="center">
    Translate PDFs with an LLM — locally for free, or through any OpenAI-compatible API. Layout and graphics preserved.
    <br />
    &middot;
    <a href="https://github.com/emirbartu/ollama-pdf-translator/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/emirbartu/ollama-pdf-translator/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## What is this

A simple but efficient Python program for translating PDFs without touching graphics. There are lots of alternatives, but they are priceyyy 🤑 and _mostly_ keep your data 🤗 — this one runs wherever you want:

- **Local Ollama** (default) — free, private, nothing leaves your machine
- **Any OpenAI-compatible API** — OpenAI, OpenRouter, LM Studio, vLLM, you name it

I was searching for worksheets for my exam, and there is a lack of resources in either English or Turkish; also, I had to keep the layout, so I created this.

It ships as both a **CLI** and a **web demo** (FastAPI, deployable to Vercel).

## Getting Started

### Prerequisites

Pick a backend:

| Backend | What you need |
|---|---|
| Local Ollama (default) | [Ollama](https://ollama.com/) installed + a model pulled, e.g. `ollama pull llama3.2` |
| OpenAI-compatible API | Base URL + API key (e.g. `https://api.openai.com/v1` + OpenAI key) |

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/emirbartu/ollama-pdf-translator.git
   cd ollama-pdf-translator
   ```
2. Install dependencies (pick one):
   ```sh
   # with uv (recommended)
   uv sync

   # or with pip
   pip install openai python-dotenv PyMuPDF tqdm fastapi "uvicorn[standard]" python-multipart jinja2 slowapi
   ```
3. (Optional) copy `.env.example` to `.env` and fill in your keys. For local Ollama you don't need to configure anything.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### CLI

**Local Ollama** (default — no config needed):

```sh
python main.py input.pdf -o translated.pdf -s English -t Spanish -m llama3.2
```

**Any OpenAI-compatible API** (e.g. OpenAI):

```sh
python main.py input.pdf -o translated.pdf -s English -t German \
  -m gpt-4o-mini --base-url https://api.openai.com/v1 --api-key sk-...
```

All options:

| Flag | Default | Description |
|---|---|---|
| `input_pdf` | — | Path to input PDF |
| `-o`, `--output` | `<name>_translated.pdf` | Output path |
| `-s`, `--source` | `English` | Source language |
| `-t`, `--target` | `Spanish` | Target language |
| `-m`, `--model` | `$LLM_MODEL` or `llama3.2` | Model name (e.g. `llama3.2`, `gpt-4o-mini`) |
| `--base-url` | auto-detect | OpenAI-compatible API base URL |
| `--api-key` | auto-detect | API key for the endpoint |
| `--skip-pages` | none | Pages to skip (0-indexed) |
| `--no-preserve-layout` | off | Don't preserve original layout |
| `--font` | `helvetica` | Fallback font |

Backend auto-detection order: `--base-url` flag → `OPENAI_BASE_URL` env → local Ollama at `http://localhost:11434/v1`.

### Web demo

A small FastAPI web UI with a language selector, drag & drop upload, and live progress:

```sh
uvicorn app:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000
```

The cloud demo runs on **visitor-provided API keys (BYOK)** — visitors point the demo at any OpenAI-compatible endpoint with their own key. Keys are used only for that request, never stored or logged. To keep the hosted demo from being abused it is deliberately limited: **max 3 pages, max 1 MB, 5 translations/hour + 20/day per IP**. The visitor model defaults to `gpt-4o-mini`.

When you run the app locally, the **/self-host** page lists your installed Ollama models (`ollama list`) and lets you pick one and translate with it directly in the browser — along with a full self-hosting guide.

Deploying to [Vercel](https://vercel.com/) works out of the box (`vercel.json` included) — no environment variables needed thanks to BYOK. Optionally set `OPENAI_API_KEY` + `OPENAI_BASE_URL` (and `LLM_MODEL`) as a server-side fallback so visitors can leave the key field empty.

### Environment variables

| Variable | Used for |
|---|---|
| `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint (overrides auto-detection) |
| `OPENAI_API_KEY` | Key for that endpoint; also the web demo's server-side fallback |
| `LLM_MODEL` | Model name used by CLI default and the web demo fallback |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## License

Distributed under the MIT licence.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Emir Bartu Ekinci - bartuekinci42@gmail.com

Project Link: [https://github.com/emirbartu/ollama-pdf-translator](https://github.com/emirbartu/ollama-pdf-translator)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
