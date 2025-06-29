# 🟩 SENDMAIL - ウェブカメラ動作検知メール通知ツール

---

> **SENDMAIL** は、ウェブカメラで動きを検知した際に  
> 指定したメールアドレスへ自動で通知を送るPythonツールです。

---

## 🚦 主な特徴
- ウェブカメラで動作を感知すると、即座にメール通知！
- 初回実行時に通知先メールアドレスとアプリパスワードを入力  
  （※自分専用で使う場合は `input(...)` 部分を変数に直接設定してOK）

---

```python
# 例：自分用に設定する場合以下を参考
#修正前
mail = input("MailAddress: ")
mail = "your@gmial.com"
#修正後
apppass = input("AppPassword: ")
apppass = "your_app_password"　　
