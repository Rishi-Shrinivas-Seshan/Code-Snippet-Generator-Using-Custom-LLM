import os
from dataclasses import dataclass, asdict
from ctransformers import AutoModelForCausalLM, AutoConfig
from flask import Flask, render_template, request, redirect, url_for


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    max_new_tokens: int
    seed: int
    reset: bool
    stream: bool
    threads: int
    stop: list[str]


def format_prompt(user_prompt: str):
    return f"""### Instruction: You are an expert coding agent. You must generate a presice code for the given task. You will be penalized for providing explanations.
{user_prompt}

### Response:"""


def generate(
    llm: AutoModelForCausalLM,
    generation_config: GenerationConfig,
    user_prompt: str,
    **kwargs,
):
    """run model inference, will return a Generator if streaming is true"""

    return llm(
        format_prompt(
            user_prompt,
        ),
        **asdict(generation_config),
    )


# Placeholder for storing generated code snippets
code_snippets = []

# Placeholder for storing given description
code_description = []


# Function to generate code snippet
def generate_code(description, new):
    if new == "True":
        generator = generate(
            llm, generation_config, description.strip(), reset="True"
        )  # For new queries, the cache is cleared so that the responses are not biased based on previous generations
    else:
        generator = generate(
            llm, generation_config, description.strip(), reset="False"
        )  # For old queries with feedback, the context needs to be retained to be able to generate refined responses

    assistant = ""
    for word in generator:
        assistant += word
    return assistant


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    snippet = None
    if request.method == "POST":
        description = request.form["description"]
        snippet = generate_code(description, new="True")
        code_description.append(description)
        code_snippets.append(snippet)
    return render_template(
        "index.html",
        snippet=snippet,
        code_snippets=code_snippets,
        code_description=code_description,
    )


@app.route("/feedback", methods=["POST"])
def feedback():
    feedback = request.form["feedback"]
    new_snippet = generate_code(feedback, new="False")
    code_description.append(feedback)
    code_snippets.append(new_snippet)
    return render_template(
        "index.html",
        snippet=new_snippet,
        code_snippets=code_snippets,
        code_description=code_description,
    )


@app.route("/view", methods=["POST"])
def view():
    snippet_index = int(request.form["snippet_index"])
    view_description = code_description[snippet_index]
    view_snippet = code_snippets[snippet_index]
    return render_template(
        "view.html",
        snippet=view_snippet,
        code_snippets=code_snippets,
        code_description=view_description,
    )


@app.route("/delete", methods=["POST"])
def delete():
    snippet_index = int(request.form["snippet_index"])
    del code_snippets[snippet_index]
    del code_description[snippet_index]
    return redirect(url_for("index"))


if __name__ == "__main__":
    config = AutoConfig.from_pretrained(
        os.path.abspath("models"),
        context_length=2048,
    )
    llm = AutoModelForCausalLM.from_pretrained(
        os.path.abspath("models/replit-v2-codeinstruct-3b.q4_1.bin"),
        model_type="replit",
        config=config,
    )

    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        max_new_tokens=512,  # adjust as needed
        seed=42,
        reset=True,  # reset history (cache)
        stream=True,  # streaming per word/token
        threads=int(os.cpu_count() / 2),  # adjust for your CPU, this will define the Computational Power to be used
        stop=["<|endoftext|>"],
    )

    app.run(debug=True)
