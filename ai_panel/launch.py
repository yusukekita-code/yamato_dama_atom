"""
AI トリオ協議システム 起動スクリプト
  1. 必要ライブラリを自動インストール
  2. ブラウザを自動で開く
  3. Flask サーバーを起動
"""
import subprocess, sys, os, time, webbrowser, threading

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

print('=' * 45)
print('  🤖 AI トリオ協議システム')
print('=' * 45)

# ── ライブラリ確認・インストール ───────────────
print('📦 ライブラリを確認中...')
try:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '-q'],
    )
    print('✅ ライブラリ OK\n')
except Exception as e:
    print(f'⚠️  インストールエラー: {e}\n')

# ── ブラウザを遅延で開く ───────────────────────
def open_browser():
    time.sleep(2.5)
    webbrowser.open('http://localhost:5000')

threading.Thread(target=open_browser, daemon=True).start()

# ── Flask 起動 ─────────────────────────────────
print('🚀 起動中 → http://localhost:5000')
print('   停止するには Ctrl+C を押してください')
print('-' * 45 + '\n')

from app import app
app.run(host='127.0.0.1', port=5000, debug=False)
