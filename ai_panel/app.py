from flask import Flask, render_template, request, jsonify
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os
import re

load_dotenv()

import urllib.request
import urllib.error
import json
import openai
import anthropic

app = Flask(__name__)

# ── クライアント初期化 ──────────────────────────────────────────────
openai_client    = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
GEMINI_API_KEY   = os.getenv('GEMINI_API_KEY', '')

OPENAI_MODEL      = os.getenv('OPENAI_MODEL',      'gpt-4o-mini')
GEMINI_MODEL      = os.getenv('GEMINI_MODEL',      'gemini-2.5-flash')
CLAUDE_MODEL      = os.getenv('CLAUDE_MODEL',      'claude-haiku-4-5-20251001')
SYNTHESIZER_MODEL = os.getenv('SYNTHESIZER_MODEL', 'gpt-4o')

BASE_SYSTEM = (
    "あなたは優秀で論理的なAIアシスタントです。回答は日本語で行ってください。"
    "回答は必ずMarkdown形式（見出し・箇条書き・**強調**など）で記述してください。"
    "プロンプトを生成・提案する場合は、プロンプト本体を <prompt></prompt> タグで囲み、"
    "タグ内の内容もMarkdown形式で整形してください。"
)

# ── AI呼び出し関数 ─────────────────────────────────────────────────
def ask_gpt(prompt: str, model: str = None) -> str:
    resp = openai_client.chat.completions.create(
        model=model or OPENAI_MODEL,
        messages=[
            {"role": "system", "content": BASE_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        max_tokens=4096
    )
    return resp.choices[0].message.content

GEMINI_FALLBACKS = [
    'gemini-2.5-flash',
    'gemini-2.5-pro',
]

_GEMINI_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}'

def _gemini_call(model_name: str, prompt: str) -> str:
    """Gemini v1beta REST API を直接呼び出す（ライブラリ非依存）"""
    url     = _GEMINI_ENDPOINT.format(model=model_name, key=GEMINI_API_KEY)
    payload = json.dumps({
        'system_instruction': {'parts': [{'text': BASE_SYSTEM}]},
        'contents':           [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig':   {'maxOutputTokens': 8192},
    }).encode('utf-8')
    req = urllib.request.Request(url, data=payload,
                                 headers={'Content-Type': 'application/json'},
                                 method='POST')
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    return data['candidates'][0]['content']['parts'][0]['text']

def ask_gemini(prompt: str) -> str:
    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    errors = []
    for model_name in models_to_try:
        try:
            return _gemini_call(model_name, prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            errors.append(f'  [{model_name}] HTTP {e.code}: {body[:300]}')
        except Exception as e:
            errors.append(f'  [{model_name}] {e}')
    raise Exception('全Geminiモデルで失敗しました:\n' + '\n'.join(errors))

def ask_claude(prompt: str) -> str:
    resp = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8192,
        system=BASE_SYSTEM,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text

def step2_5_facilitate(question: str, answers: dict) -> dict:
    """ファシリテーター分析: 合意点・対立点・完成度スコアを返す"""
    ai_names = {'chatgpt': 'ChatGPT', 'gemini': 'Gemini', 'claude': 'Claude'}
    valid   = {k: v for k, v in answers.items() if not v.startswith('⚠️ エラー:')}
    errored = [k for k in answers if k not in valid]

    if not valid:
        return {'report': '⚠️ 全AIがエラーのため分析できません。', 'score': 0, 'should_extend': True}

    answer_text = '\n\n'.join(f'[{ai_names.get(k, k)}]\n{v}' for k, v in valid.items())
    error_note  = (
        f'\n\n※ {", ".join(ai_names.get(k, k) for k in errored)} はエラーのため回答なし。'
        if errored else ''
    )

    prompt = (
        f'あなたは議論のファシリテーターです。以下の質問に対する各AIの回答を分析してください。\n\n'
        f'質問: {question}{error_note}\n\n'
        f'=== 各AIの回答 ===\n{answer_text}\n\n'
        f'以下の形式で出力してください：\n\n'
        f'## 合意点\n（各AIが共通して述べている重要なポイントを箇条書きで）\n\n'
        f'## 対立点・見解の相違\n（意見や強調点が異なる部分を具体的に記述。なければ「特になし」）\n\n'
        f'## 未カバーの重要な疑問\n（議論で答えられていない問いや見落としを箇条書きで）\n\n'
        f'## ファシリテーター所見\n（議論の評価とStep2での改善提案）\n\n'
        f'## 完成度スコア: XX点\n'
        f'（0〜100点。採点基準：質問への網羅性30点、具体性・根拠30点、相互補完性20点、実用的結論20点）'
    )

    report = ask_gpt(prompt, model=SYNTHESIZER_MODEL)
    score  = 50
    m = re.search(r'完成度スコア[：:]\s*(\d+)', report)
    if m:
        score = min(100, max(0, int(m.group(1))))

    return {'report': report, 'score': score, 'should_extend': score < 70}


CALLERS = {
    'chatgpt': ask_gpt,
    'gemini':  ask_gemini,
    'claude':  ask_claude,
}

def run_parallel(prompts: dict) -> dict:
    """prompts = {ai_name: prompt}  →  {ai_name: response}"""
    out = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(CALLERS[n], p): n for n, p in prompts.items()}
        for f in as_completed(futs):
            n = futs[f]
            try:
                out[n] = f.result()
            except Exception as e:
                out[n] = f"⚠️ エラー: {e}"
    return out

# ── ルート ─────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test-connection', methods=['GET'])
def test_connection():
    """各AIへの接続テスト（短いプロンプトで確認）"""
    test_prompt = '「OK」とだけ返してください。'
    results = {}

    # ChatGPT
    try:
        ask_gpt(test_prompt)
        results['chatgpt'] = {'ok': True, 'model': OPENAI_MODEL}
    except Exception as e:
        results['chatgpt'] = {'ok': False, 'error': str(e)[:300]}

    # Gemini
    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    gemini_ok = False
    for model_name in models_to_try:
        try:
            _gemini_call(model_name, test_prompt)
            results['gemini'] = {'ok': True, 'model': model_name}
            gemini_ok = True
            break
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', errors='replace')
            results['gemini'] = {'ok': False, 'error': f'HTTP {e.code}: {body[:300]}', 'tried': model_name}
        except Exception as e:
            results['gemini'] = {'ok': False, 'error': str(e)[:300], 'tried': model_name}
    if not gemini_ok and 'gemini' not in results:
        results['gemini'] = {'ok': False, 'error': '全モデル失敗'}

    # Claude
    try:
        ask_claude(test_prompt)
        results['claude'] = {'ok': True, 'model': CLAUDE_MODEL}
    except Exception as e:
        results['claude'] = {'ok': False, 'error': str(e)[:300]}

    return jsonify(results)


@app.route('/api/load-self', methods=['GET'])
def load_self():
    """このシステム自身のソースコードを返す"""
    base = os.path.dirname(os.path.abspath(__file__))
    files = {}
    targets = [('app.py', 'app.py'), ('index.html', 'templates/index.html')]
    for name, rel in targets:
        try:
            with open(os.path.join(base, rel), encoding='utf-8') as f:
                files[name] = f.read()
        except Exception as e:
            files[name] = f'読み込みエラー: {e}'
    return jsonify(files)

@app.route('/api/step1', methods=['POST'])
def step1():
    data = request.json
    q   = data['question']
    ctx = data.get('context', '').strip()
    ctx_part = f"\n\n【参考資料】\n{ctx}" if ctx else ""
    prompt = (
        f"以下の質問に詳しく回答してください。\n"
        f"・主要なポイントを明確に述べること\n"
        f"・具体例や根拠を含めること\n"
        f"・追加情報が必要な場合は【質問】として明記すること"
        f"{ctx_part}\n\n"
        f"質問: {q}"
    )
    return jsonify(run_parallel({k: prompt for k in CALLERS}))

@app.route('/api/step2', methods=['POST'])
def step2():
    data = request.json
    q, s1 = data['question'], data['step1']
    ctx = data.get('context', '').strip()
    fac = data.get('facilitate_report', '').strip()
    ctx_part = f"\n【参考資料】\n{ctx}\n" if ctx else ""
    fac_part = f"\n【ファシリテーター分析（改善の参考に）】\n{fac}\n" if fac else ""

    def make_prompt(my_name):
        others = "\n\n".join(
            f"【{n} の回答】\n{r}" for n, r in s1.items() if n != my_name
        )
        return (
            f"元の質問: {q}\n"
            f"{ctx_part}\n"
            f"【あなたの初回回答】\n{s1[my_name]}\n\n"
            f"【他のAIの回答】\n{others}\n\n"
            f"{fac_part}"
            f"上記を踏まえて以下を行ってください：\n"
            f"1. 他のAIの回答で評価できる点・参考になる点\n"
            f"2. あなたの回答の補足・修正事項\n"
            f"3. まだカバーされていない重要な観点の追加\n"
            f"4. 残る疑問点があれば【質問】として明記"
        )

    return jsonify(run_parallel({k: make_prompt(k) for k in CALLERS}))


@app.route('/api/step2_5', methods=['POST'])
def step2_5():
    data   = request.json
    result = step2_5_facilitate(data['question'], data['step1'])
    return jsonify(result)

@app.route('/api/step3', methods=['POST'])
def step3():
    data = request.json
    q, s1, s2 = data['question'], data['step1'], data['step2']

    prompt = f"""質問: {q}

=== 各AIの初回回答 ===
[ChatGPT]
{s1.get('chatgpt','')}

[Gemini]
{s1.get('gemini','')}

[Claude]
{s1.get('claude','')}

=== 各AIの相互レビュー ===
[ChatGPT レビュー]
{s2.get('chatgpt','')}

[Gemini レビュー]
{s2.get('gemini','')}

[Claude レビュー]
{s2.get('claude','')}

【出力ルール】
- 上記各AIの回答はMarkdown形式で記述されている。最終統合回答も必ずMarkdown形式で出力すること
- 各AIが使用した見出し・箇条書き・強調などの書式を参考にし、同等以上の構造と詳細度で記述すること
- プレーンテキストへの変換は禁止。見出し（##/###）・箇条書き（-/1.）・**強調**を積極的に使用すること

以上をすべて統合して、以下の形式で出力してください：

## 最終統合回答
（各AIの回答を参考にしたMarkdown形式での包括的・完全な回答）

## 各AIの貢献分析

### ChatGPT
- 独自の観点: （特徴的な貢献を具体的に記述）
- 全体評価: （S／A／B／C）

### Gemini
- 独自の観点: （特徴的な貢献を具体的に記述）
- 全体評価: （S／A／B／C）

### Claude
- 独自の観点: （特徴的な貢献を具体的に記述）
- 全体評価: （S／A／B／C）

## 未解決の重要な質問
（各AIが提起した中で未回答の重要な疑問点があれば列挙）"""

    # Step3はより高品質なモデルで統合
    result = ask_gpt(prompt, model=SYNTHESIZER_MODEL)
    return jsonify({'result': result})

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    os._exit(0)

if __name__ == '__main__':
    app.run(debug=False, port=5000)
