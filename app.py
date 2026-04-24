"""
StudyMate v6.0 — 完整后端
核心升级：
  - AI 升级（llama3-70b + 完整上下文 + 学习专属 prompt）
  - 行为感知专注度（鼠标/键盘/空闲/标签切换）
  - 个性化学习建模（学习画像 + 智能建议 + 最佳时段分析）
  - 100条模拟排名数据
  - 完整TTS提醒接口
  - 登录流程修复
"""
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import sqlite3, hashlib, time, os, uuid, random, json, base64, math
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH    = os.environ.get("DB_PATH",    os.path.join(BASE_DIR, "studymate.db"))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
SECRET_KEY = os.environ.get("SECRET_KEY", "studymate_2024_key")

# ── Groq API Key（内置，也可通过环境变量覆盖）
GROQ_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_KEY:
    raise ValueError("GROQ_API_KEY is not set in environment variables")

for sub in ["avatars", "rank_bgs", "posts", ""]:
    os.makedirs(os.path.join(UPLOAD_DIR, sub), exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# AI ENGINE — llama3-70b（Groq 免费，中文效果最好）
# 备选免费 AI：
#   Google Gemini:  pip install google-generativeai，申请 AIStudio Key
#   Moonshot (Kimi): 国内可用，每月免费额度
#   智谱 GLM-4:      国内注册免费额度
# ══════════════════════════════════════════════════════════════════
_groq_client = None

def get_groq():
    global _groq_client
    if _groq_client is None and GROQ_KEY:
        try:
            from groq import Groq
            _groq_client = Groq(api_key=GROQ_KEY)
            print("✅ Groq (llama3-70b) 已连接")
        except Exception as e:
            print(f"❌ Groq 初始化失败: {e}")
    return _groq_client

# 按优先级尝试不同模型
# Groq 可用模型（2025年，按优先级排列）
# 完整最新列表见 https://console.groq.com/docs/models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",      # 最新最强，中文好（替代 llama3-70b-8192）
    "llama-3.1-70b-versatile",      # 备选
    "llama3-70b-8192",              # 已下线，保留作最后尝试
    "mixtral-8x7b-32768",           # 混合专家模型，中文可用
    "llama-3.1-8b-instant",         # 轻量快速
    "llama3-8b-8192",               # 最小模型，最后备选
    "gemma2-9b-it",                 # Google Gemma 备选
]

def groq_chat(messages, max_tokens=200, temperature=0.78):
    """调用 Groq，自动降级到下一个可用模型"""
    g = get_groq()
    if not g:
        return None
    skip_keywords = ["decommissioned", "not found", "does not exist", "deprecated"]
    for model in GROQ_MODELS:
        try:
            r = g.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=temperature,
            )
            text = r.choices[0].message.content.strip()
            # 记录第一个成功的模型（避免每次重试）
            if not hasattr(groq_chat, '_working_model') or groq_chat._working_model != model:
                groq_chat._working_model = model
                print(f"✅ 使用模型: {model}")
            return text
        except Exception as e:
            err_str = str(e).lower()
            if any(kw in err_str for kw in skip_keywords):
                print(f"⏭ 模型 {model} 已下线，尝试下一个...")
                continue
            # 其他错误（网络、速率限制等）直接返回 None
            print(f"❌ Groq {model} error: {e}")
            return None
    print("❌ 所有 Groq 模型均不可用")
    return None

# ── 小灯系统提示词（精心设计，避免答非所问）
LAMP_SYSTEM_PROMPT = """你是「小灯」，一个聪明有趣的学习陪伴AI助手。

【性格设定】
- 像智慧的好友，说话自然口语，偶尔幽默
- 简洁有力，不废话，不说大道理
- 真正理解用户，给出具体可操作的建议
- 学习相关问题给专业建议，闲聊轻松回应

【回复规则】
- 必须用中文回复
- 每次回复控制在60字以内
- 如果用户问问题，认真回答问题本身
- 如果用户说学习困难，给出学习方法建议
- 如果用户分心，温和提醒专注
- 不要每次都说"加油"，要有变化

【禁止行为】
- 不重复用户说过的话
- 不用"我理解你的感受"这类套话
- 不一直说鼓励话而不回答实际问题"""

def lamp_reply(user_msg, context="", history=None):
    """小灯对话，带历史上下文"""
    messages = [{"role": "system", "content": LAMP_SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"[用户当前状态] {context}"})
    # 带入最近4轮历史
    if history:
        for h in history[-4:]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_msg})
    return groq_chat(messages, max_tokens=150, temperature=0.8)

def generate_remind_msg(username, plan_title, elapsed_min, total_today_min):
    """生成定时提醒话语"""
    hour = datetime.now().hour
    time_map = {range(5,9):"早上好",range(9,12):"上午好",
                range(12,14):"中午了",range(14,17):"下午好",
                range(17,19):"傍晚了",range(19,22):"晚上好"}
    time_ctx = next((v for k,v in time_map.items() if hour in k), "深夜了")

    prompt = (
        f"你是学习AI「小灯」，给用户{username}发一句{time_ctx}学习提醒。"
        f"他正在学「{plan_title}」，已学{elapsed_min:.0f}分钟，今天累计{total_today_min:.0f}分钟。"
        f"写一句话（15字内，口语化，可以俏皮），只输出这句话。"
    )
    result = groq_chat([{"role": "user", "content": prompt}], max_tokens=50, temperature=0.9)
    fallbacks = [
        f"已学{elapsed_min:.0f}分钟了，坚持一下 💪",
        f"「{plan_title}」进行中，继续保持！",
        f"今天已学{total_today_min:.0f}分钟，棒棒的！",
    ]
    return result or random.choice(fallbacks)

def generate_decision_msg(action, focus, dur, username="你"):
    """AI决策提醒"""
    prompts = {
        "alert_away":   f"用户离开屏幕了，写一句温和催促（10字内中文）",
        "force_rest":   f"用户学了{dur:.0f}分钟，写一句建议休息（10字内中文）",
        "suggest_rest": f"用户有点累，写温柔的建议（10字内中文）",
        "remind_focus": f"用户专注度下降到{focus:.0%}，写提醒专注（10字内中文，有力量）",
        "praise":       f"用户专注度{focus:.0%}坚持了{dur:.0f}分钟，写表扬（12字内中文）",
        "encourage":    f"用户在努力学习，写鼓励（10字内中文，有变化）",
    }
    fallbacks = {
        "alert_away": "回来继续吧～",
        "force_rest": f"学了{dur:.0f}分钟了，休息一下",
        "suggest_rest": "累了就歇歇 ☕",
        "remind_focus": "专注拉回来！",
        "praise": f"太棒了！专注度{focus:.0%} 🌟",
        "encourage": "稳住，你在进步 💫",
    }
    p = prompts.get(action, prompts["encourage"])
    result = groq_chat([{"role":"user","content":p+"，只输出这句话。"}], max_tokens=40, temperature=0.88)
    return result or fallbacks.get(action, "继续加油！")

def generate_learning_advice(profile: dict, username: str) -> str:
    """基于学习画像生成个性化建议"""
    best_hour = profile.get("best_hour", 9)
    avg_session = profile.get("avg_session_min", 45)
    focus_avg = profile.get("focus_avg", 0.65)
    streak = profile.get("streak_days", 0)
    weak_hours = profile.get("weak_hours", [])

    prompt = (
        f"你是学习顾问「小灯」，为{username}提供个性化学习建议。"
        f"数据：最佳学习时段{best_hour}点，平均专注度{focus_avg:.0%}，"
        f"平均每次学习{avg_session:.0f}分钟，连续打卡{streak}天，"
        f"低效时段{weak_hours}。"
        f"写3条具体建议（每条不超过25字，编号1.2.3.），直接给建议不要解释。"
    )
    result = groq_chat([{"role": "user", "content": prompt}], max_tokens=200, temperature=0.7)
    return result or f"1. 每天在{best_hour}点前后学习效果最佳\n2. 保持{avg_session:.0f}分钟的专注时段\n3. 连续打卡{streak}天，继续坚持！"

# ══════════════════════════════════════════════════════════════════
# 数据库
# ══════════════════════════════════════════════════════════════════
def get_db():
    uri = DB_PATH.startswith("file:")
    conn = sqlite3.connect(DB_PATH, uri=uri)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

# ── 模拟用户名池
MOCK_NAMES = [
    "学霸小王","专注达人","晨读少年","夜深学长","题海游泳者",
    "数学小神","英语达人","编程少年","化学实验家","历史探索者",
    "自律同学","破万卷书","笔记控阿明","图书馆常客","努力的李华",
    "地理向导","生物小博士","物理爱好者","文学小才女","哲学思考者",
    "用功小陈","勤奋小张","专注阿华","早起达人","晚学能手",
    "博览群书","规律作息","精华笔记","刷题达人","借书王者",
    "代码战士","算法达人","微积分控","有机化学迷","量子物理迷",
    "古文爱好者","诗词背诵达人","英语口语达人","数学竞赛选手","物理竞赛冠军",
    "每日打卡者","计划执行者","时间管理达人","番茄钟爱好者","深度学习者",
    "思维导图控","费曼学习法","主动回忆者","间隔复习达人","高效笔记者",
] + [f"学习者{i:03d}" for i in range(51, 101)]

def init_db():
    with get_db() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            avatar_url TEXT DEFAULT '',
            avatar_data TEXT DEFAULT '',
            bio TEXT DEFAULT '',
            lamp_skin TEXT DEFAULT 'default',
            lamp_remind_interval INTEGER DEFAULT 30,
            lamp_remind_enabled INTEGER DEFAULT 1,
            join_ranking INTEGER DEFAULT 1,
            rank_bg_url TEXT DEFAULT '',
            is_mock INTEGER DEFAULT 0,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS user_prefs(
            user_id TEXT PRIMARY KEY,
            bg_scene TEXT DEFAULT 'library',
            bg_custom_url TEXT DEFAULT '',
            daily_goal_min INTEGER DEFAULT 60
        );
        CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL,
            duration_min REAL DEFAULT 0,
            avg_focus REAL DEFAULT 0,
            pomodoros INTEGER DEFAULT 0,
            plan_id TEXT DEFAULT NULL,
            hour_of_day INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS focus_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            focus_score REAL NOT NULL,
            fatigue REAL DEFAULT 0,
            is_away INTEGER DEFAULT 0,
            source TEXT DEFAULT 'behavior'
        );
        CREATE TABLE IF NOT EXISTS behavior_log(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            event_type TEXT NOT NULL,
            data TEXT DEFAULT '{}',
            focus_contribution REAL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS study_plans(
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            duration_min INTEGER DEFAULT 25,
            subject TEXT DEFAULT '',
            color TEXT DEFAULT '#c49a3c',
            is_completed INTEGER DEFAULT 0,
            sort_order INTEGER DEFAULT 0,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS checkins(
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT DEFAULT NULL,
            photo_path TEXT DEFAULT '',
            note TEXT DEFAULT '',
            is_public INTEGER DEFAULT 1,
            likes INTEGER DEFAULT 0,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS quotes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            author TEXT DEFAULT '',
            category TEXT DEFAULT 'general',
            user_id TEXT DEFAULT NULL,
            is_custom INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS posts(
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            image_url TEXT DEFAULT '',
            likes INTEGER DEFAULT 0,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS comments(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS post_likes(
            post_id TEXT, user_id TEXT, PRIMARY KEY(post_id,user_id)
        );
        CREATE TABLE IF NOT EXISTS user_rewards(
            user_id TEXT PRIMARY KEY,
            total_minutes INTEGER DEFAULT 0,
            unlocked_skins TEXT DEFAULT '["default"]'
        );
        CREATE TABLE IF NOT EXISTS chat_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL DEFAULT(unixepoch())
        );
        CREATE TABLE IF NOT EXISTS learning_profile(
            user_id TEXT PRIMARY KEY,
            best_hour INTEGER DEFAULT 9,
            avg_session_min REAL DEFAULT 45,
            focus_avg REAL DEFAULT 0.65,
            streak_days INTEGER DEFAULT 0,
            last_study_date TEXT DEFAULT '',
            total_sessions INTEGER DEFAULT 0,
            total_days INTEGER DEFAULT 0,
            weak_hours TEXT DEFAULT '[]',
            strong_subjects TEXT DEFAULT '[]',
            pref_session_type TEXT DEFAULT 'short',
            last_updated REAL DEFAULT 0
        );
        """)

        # 内置语录
        if c.execute("SELECT COUNT(*) FROM quotes WHERE is_custom=0").fetchone()[0] == 0:
            c.executemany("INSERT INTO quotes(text,author,category) VALUES(?,?,?)", [
                ("学而不思则罔，思而不学则殆。","孔子","study"),
                ("天才是百分之一的灵感，加上百分之九十九的汗水。","爱迪生","motivate"),
                ("读书不觉已春深，一寸光阴一寸金。","王贞白","study"),
                ("不积跬步，无以至千里；不积小流，无以成江海。","荀子","study"),
                ("路漫漫其修远兮，吾将上下而求索。","屈原","motivate"),
                ("知之者不如好之者，好之者不如乐之者。","孔子","study"),
                ("书山有路勤为径，学海无涯苦作舟。","韩愈","study"),
                ("业精于勤荒于嬉，行成于思毁于随。","韩愈","motivate"),
                ("每一个不曾起舞的日子，都是对生命的辜负。","尼采","life"),
                ("你若盛开，清风自来。","仓央嘉措","life"),
                ("专注是一种超能力。","佚名","focus"),
                ("今天多学一点，明天少迷茫一点。","佚名","study"),
                ("别在该努力的年纪选择了安逸。","佚名","motivate"),
                ("你现在的努力，是为了让未来的自己感谢现在的自己。","佚名","motivate"),
                ("休息是为了走更长的路。","佚名","rest"),
                ("把大目标拆成小任务，一步一步来。","佚名","focus"),
                ("每次番茄都是一次小胜利。","佚名","focus"),
                ("深呼吸，继续。","佚名","rest"),
                ("愿你出走半生，归来仍是少年。","民间","life"),
                ("自律给你自由。","艾比克泰德","focus"),
                ("当你感到疲惫，说明你在走上坡路。","佚名","motivate"),
                ("人生没有捷径，每一步都算数。","佚名","life"),
                ("成功是每天重复做该做的事。","亚里士多德","motivate"),
                ("读书破万卷，下笔如有神。","杜甫","study"),
                ("不要等待完美的时机，现在就开始。","佚名","motivate"),
                ("学习的本质是改变自己的思考方式。","佚名","study"),
                ("专注当下，就是最好的学习状态。","佚名","focus"),
                ("坚持是最好的天赋。","佚名","motivate"),
                ("静下心来，才能做好一件事。","佚名","focus"),
                ("今日事今日毕，明日还有明日事。","佚名","study"),
            ])

        # 生成100个模拟用户（排名展示）
        mock_count = c.execute("SELECT COUNT(*) FROM users WHERE is_mock=1").fetchone()[0]
        if mock_count < 90:
            _seed_mock_users(c)

        # 示例社区帖子
        if c.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == 0:
            demo_uid = str(uuid.uuid4())
            c.execute("INSERT OR IGNORE INTO users(id,username,password,bio) VALUES(?,?,?,?)",
                (demo_uid,"示例用户",hashlib.sha256("demo".encode()).hexdigest(),"热爱学习的同学 📚"))
            c.execute("INSERT OR IGNORE INTO user_prefs(user_id) VALUES(?)",(demo_uid,))
            c.execute("INSERT OR IGNORE INTO user_rewards(user_id) VALUES(?)",(demo_uid,))
            for content,likes,offset in [
                ("今天番茄钟学了4小时高数，专注度创新高 🍅 推荐大家试试！",18,3600),
                ("英语单词打卡第14天，坚持就是胜利 💪 每天30个单词",12,7200),
                ("费曼学习法真的有用！把学过的知识讲给别人听，立刻发现哪里没搞懂 ✨",25,86400),
                ("分享学习时间表：7-9点数学，10-12点英语，下午编程。按照生物钟安排效率翻倍！",32,172800),
            ]:
                c.execute("INSERT INTO posts(id,user_id,content,likes,created_at) VALUES(?,?,?,?,?)",
                    (str(uuid.uuid4()),demo_uid,content,likes,time.time()-offset))
        c.commit()
        _seed_user_456(c)
        c.commit()
    _migrate()

def _seed_user_456(c):
    """预设演示账号 456 / 密码 123456，含30天丰富学习数据"""
    existing = c.execute("SELECT id FROM users WHERE username='456'").fetchone()
    if existing:
        return
    DEMO_UID = "demo-user-456-preset-0000000001"
    pw_hash = hashlib.sha256(("123456" + SECRET_KEY).encode()).hexdigest()
    c.execute(
        "INSERT OR IGNORE INTO users(id,username,password,bio,is_mock,join_ranking) VALUES(?,?,?,?,0,1)",
        (DEMO_UID, "456", pw_hash, "每天打卡学习，记录成长轨迹 📚")
    )
    c.execute("INSERT OR IGNORE INTO user_prefs(user_id,daily_goal_min) VALUES(?,?)", (DEMO_UID, 90))
    c.execute("INSERT OR IGNORE INTO user_rewards(user_id,total_minutes,unlocked_skins) VALUES(?,?,?)",
              (DEMO_UID, 3260, '["default","cozy","pro"]'))
    # 学习计划
    plans_data = [
        ("高数微积分", "数学",  50, "#c49a3c"),
        ("英语四级精读","英语",  25, "#5c7a5c"),
        ("Python编程",  "编程",  60, "#4a7a9b"),
        ("物理力学",   "物理",  45, "#6b5b95"),
    ]
    for i, (title, subj, dur, color) in enumerate(plans_data):
        c.execute(
            "INSERT OR IGNORE INTO study_plans(id,user_id,title,subject,duration_min,color,sort_order) VALUES(?,?,?,?,?,?,?)",
            (f"plan456_{i}", DEMO_UID, title, subj, dur, color, i)
        )
    # 30天学习记录（持续成长曲线）
    now = time.time()
    rng = random.Random(456)  # 固定种子，保证数据一致性
    SUBJECTS = ["数学","数学","英语","编程","物理","英语","编程"]
    NOTES = [
        "今天完成了今日学习目标！✅",
        "专注度创新高，番茄工作法真的有效 🍅",
        "坚持就是胜利 💪",
        "费曼学习法真的管用！",
        "每天进步一点点，慢慢就会有大变化",
        "英语阅读速度明显提升了",
        "编程题全对，开心！",
        "加油，离目标越来越近了",
    ]
    checkin_count = 0
    for day in range(30, 0, -1):
        if rng.random() < 0.12:   # 88% 出勤率
            continue
        # 学习时长随时间递增（模拟成长）
        if day > 20:
            base_min = rng.uniform(35, 75)
        elif day > 10:
            base_min = rng.uniform(65, 130)
        else:
            base_min = rng.uniform(90, 180)
        # 时段：早上9点或晚上20点
        hour = rng.choice([8, 9, 10, 14, 19, 20, 21])
        day_base = now - day * 86400
        start_t = day_base + hour * 3600 + rng.uniform(0, 2400)
        focus = rng.uniform(0.64, 0.93)
        # 后期专注度更高
        if day < 10:
            focus = min(0.95, focus + 0.06)
        pomos = max(1, int(base_min / 25))
        subj = rng.choice(SUBJECTS)
        sess_id = f"d456_s{day}"
        # 找到对应计划ID
        plan_map = {"数学": "plan456_0", "英语": "plan456_1", "编程": "plan456_2", "物理": "plan456_3"}
        plan_id = plan_map.get(subj)
        c.execute("""INSERT OR IGNORE INTO sessions
            (id,user_id,start_time,end_time,duration_min,avg_focus,pomodoros,plan_id,hour_of_day)
            VALUES(?,?,?,?,?,?,?,?,?)""",
            (sess_id, DEMO_UID, start_t, start_t + base_min * 60,
             round(base_min, 1), round(focus, 3), pomos, plan_id, hour))
        # 打卡（约72%的学习日）
        if rng.random() < 0.72:
            c.execute("""INSERT OR IGNORE INTO checkins
                (id,user_id,session_id,note,is_public,created_at)
                VALUES(?,?,?,?,1,?)""",
                (f"d456_c{day}", DEMO_UID, sess_id, rng.choice(NOTES), start_t + base_min * 60))
            checkin_count += 1
    # 学习画像
    c.execute("""INSERT OR IGNORE INTO learning_profile
        (user_id,best_hour,avg_session_min,focus_avg,streak_days,last_study_date,
         total_sessions,total_days,weak_hours,strong_subjects,pref_session_type,last_updated)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
        (DEMO_UID, 9, 78.5, 0.79, 14,
         datetime.now().strftime("%Y-%m-%d"), 26, 25,
         "[13,14,22]", '["数学","编程"]', "short", time.time()))

def _seed_mock_users(c):
    """生成100个模拟用户 + 学习数据（排名演示）"""
    now = time.time()
    for i, name in enumerate(MOCK_NAMES[:100]):
        mid = f"mock_{i:04d}"
        try:
            c.execute("INSERT OR IGNORE INTO users(id,username,password,is_mock,join_ranking) VALUES(?,?,?,1,1)",
                      (mid, name, "mock"))
            c.execute("INSERT OR IGNORE INTO user_prefs(user_id) VALUES(?)", (mid,))
            c.execute("INSERT OR IGNORE INTO user_rewards(user_id) VALUES(?)", (mid,))
        except: pass

        # 指数分布：少数高手学很多，大多数中等
        if i < 3:      base_min = random.uniform(240, 420)   # 4-7小时
        elif i < 10:   base_min = random.uniform(120, 240)   # 2-4小时
        elif i < 30:   base_min = random.uniform(60, 120)    # 1-2小时
        elif i < 60:   base_min = random.uniform(25, 60)     # 25-60分钟
        else:           base_min = random.uniform(5, 25)     # 5-25分钟

        # 今日数据
        af = random.uniform(0.55, 0.95)
        pomo = max(1, int(base_min / 25))
        start_t = now - random.uniform(3600, 72000)
        try:
            c.execute("""INSERT OR IGNORE INTO sessions(id,user_id,start_time,end_time,duration_min,avg_focus,pomodoros)
                VALUES(?,?,?,?,?,?,?)""",
                (f"ms_d_{mid}", mid, start_t, start_t+base_min*60, round(base_min,1), round(af,3), pomo))
        except: pass

        # 本周数据（7天）
        for day in range(1, 7):
            w_min = base_min * random.uniform(0.5, 1.4)
            ws = now - day*86400 + random.uniform(3600, 75600)
            try:
                c.execute("""INSERT OR IGNORE INTO sessions(id,user_id,start_time,end_time,duration_min,avg_focus,pomodoros)
                    VALUES(?,?,?,?,?,?,?)""",
                    (f"ms_w{day}_{mid}", mid, ws, ws+w_min*60, round(w_min,1),
                     round(random.uniform(0.5,0.95),3), max(1,int(w_min/25))))
            except: pass

        # 本月数据（30天）
        for day in range(7, 30):
            if random.random() < 0.65:  # 65%出勤率
                m_min = base_min * random.uniform(0.4, 1.3)
                ms2 = now - day*86400 + random.uniform(3600, 75600)
                try:
                    c.execute("""INSERT OR IGNORE INTO sessions(id,user_id,start_time,end_time,duration_min,avg_focus,pomodoros)
                        VALUES(?,?,?,?,?,?,?)""",
                        (f"ms_m{day}_{mid}", mid, ms2, ms2+m_min*60, round(m_min,1),
                         round(random.uniform(0.45,0.92),3), max(1,int(m_min/25))))
                except: pass

def _migrate():
    cols = [
        ("users","avatar_data","TEXT DEFAULT ''"),
        ("users","lamp_remind_interval","INTEGER DEFAULT 30"),
        ("users","lamp_remind_enabled","INTEGER DEFAULT 1"),
        ("users","join_ranking","INTEGER DEFAULT 1"),
        ("users","rank_bg_url","TEXT DEFAULT ''"),
        ("users","lamp_skin","TEXT DEFAULT 'default'"),
        ("users","is_mock","INTEGER DEFAULT 0"),
        ("sessions","hour_of_day","INTEGER DEFAULT 0"),
        ("focus_log","source","TEXT DEFAULT 'behavior'"),
    ]
    with get_db() as c:
        for tbl,col,typ in cols:
            try: c.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {typ}"); c.commit()
            except: pass

init_db()

# ══════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════
def hp(pw): return hashlib.sha256((pw+SECRET_KEY).encode()).hexdigest()
def mk_tok(uid): return hashlib.sha256(f"{uid}:{time.time()}:{SECRET_KEY}".encode()).hexdigest()[:32]+uid

def auth_req(f):
    @wraps(f)
    def w(*a,**k):
        tok = request.headers.get("Authorization","").replace("Bearer ","")
        if not tok or len(tok)<32: return jsonify({"error":"未登录"}),401
        uid = tok[32:]
        with get_db() as db:
            u = db.execute("SELECT * FROM users WHERE id=? AND is_mock=0",(uid,)).fetchone()
        if not u: return jsonify({"error":"无效令牌"}),401
        request.user = dict(u)
        return f(*a,**k)
    return w

ok  = lambda d=None,**k: jsonify({"ok":True,"data":d,**k})
err = lambda m,c=400: (jsonify({"ok":False,"error":m}),c)

@app.get("/")
def index(): return send_file(os.path.join(BASE_DIR,"index.html"))
@app.get("/uploads/<path:fn>")
def serve_upload(fn): return send_from_directory(UPLOAD_DIR,fn)

# ══════════════════════════════════════════════════════════════════
# 认证 — 修复登录流程
# ══════════════════════════════════════════════════════════════════
@app.post("/api/register")
def register():
    d = request.json or {}
    u, p = d.get("username","").strip(), d.get("password","")
    if not u or not p: return err("用户名和密码不能为空")
    if not(2<=len(u)<=20): return err("用户名2-20位")
    if len(p)<6: return err("密码至少6位")
    if u in MOCK_NAMES: return err("该用户名已被使用，请换一个")
    uid = str(uuid.uuid4())
    try:
        with get_db() as db:
            db.execute("INSERT INTO users(id,username,password) VALUES(?,?,?)",(uid,u,hp(p)))
            db.execute("INSERT INTO user_prefs(user_id) VALUES(?)",(uid,))
            db.execute("INSERT INTO user_rewards(user_id) VALUES(?)",(uid,))
            db.execute("INSERT INTO learning_profile(user_id) VALUES(?)",(uid,))
            for i,(title,subj,dur,color) in enumerate([
                ("高数复习","数学",50,"#c49a3c"),
                ("英语单词","英语",25,"#5c7a5c"),
                ("编程练习","编程",60,"#4a7a9b"),
            ]):
                db.execute("INSERT INTO study_plans(id,user_id,title,subject,duration_min,color,sort_order) VALUES(?,?,?,?,?,?,?)",
                    (str(uuid.uuid4()),uid,title,subj,dur,color,i))
            db.commit()
    except sqlite3.IntegrityError:
        return err("用户名已存在")
    token = mk_tok(uid)
    # 注册成功直接返回 token，前端可直接登录
    return ok({"token":token,"user_id":uid,"username":u,"is_new":True})

@app.post("/api/login")
def login():
    d = request.json or {}
    with get_db() as db:
        u = db.execute("SELECT * FROM users WHERE username=? AND password=? AND is_mock=0",
                       (d.get("username",""), hp(d.get("password","")))).fetchone()
    if not u: return err("用户名或密码错误",401)
    token = mk_tok(u["id"])
    return ok({"token":token,"user_id":u["id"],"username":u["username"],"is_new":False})

@app.get("/api/me")
@auth_req
def me():
    u = request.user
    with get_db() as db:
        s = db.execute("""SELECT COUNT(*) s,COALESCE(SUM(duration_min),0) m,
            COALESCE(AVG(avg_focus),0) f,COALESCE(SUM(pomodoros),0) p
            FROM sessions WHERE user_id=? AND end_time IS NOT NULL""",(u["id"],)).fetchone()
        chk = db.execute("SELECT COUNT(*) n FROM checkins WHERE user_id=?",(u["id"],)).fetchone()
        prefs = db.execute("SELECT * FROM user_prefs WHERE user_id=?",(u["id"],)).fetchone()
        rw = db.execute("SELECT * FROM user_rewards WHERE user_id=?",(u["id"],)).fetchone()
        lp = db.execute("SELECT * FROM learning_profile WHERE user_id=?",(u["id"],)).fetchone()
    return ok({
        "user_id":u["id"],"username":u["username"],
        "avatar_url":u["avatar_url"],"avatar_data":u.get("avatar_data",""),
        "bio":u.get("bio",""),"lamp_skin":u.get("lamp_skin","default"),
        "lamp_remind_interval":u.get("lamp_remind_interval",30),
        "lamp_remind_enabled":u.get("lamp_remind_enabled",1),
        "join_ranking":u.get("join_ranking",1),
        "stats":{"sessions":s["s"],"total_min":round(s["m"] or 0,1),
                 "avg_focus":round((s["f"] or 0)*100,1),"checkins":chk["n"],"pomodoros":s["p"]},
        "prefs":dict(prefs) if prefs else {},
        "rewards":{"total_minutes":rw["total_minutes"] if rw else 0,
                   "unlocked_skins":json.loads(rw["unlocked_skins"]) if rw else ["default"]},
        "learning_profile":dict(lp) if lp else {},
    })

@app.put("/api/me")
@auth_req
def update_me():
    d = request.json or {}; uid = request.user["id"]
    allowed = ["bio","lamp_skin","lamp_remind_interval","lamp_remind_enabled","join_ranking"]
    with get_db() as db:
        for k in allowed:
            if k in d:
                val = str(d[k])[:200] if k=="bio" else d[k]
                db.execute(f"UPDATE users SET {k}=? WHERE id=?",(val,uid))
        db.commit()
    return ok()

@app.post("/api/me/avatar")
@auth_req
def upload_avatar():
    uid = request.user["id"]
    f = request.files.get("avatar")
    if f and f.filename:
        ext = f.filename.rsplit(".",1)[-1].lower()
        if ext not in("jpg","jpeg","png","webp","gif"): return err("不支持的格式")
        fname = f"{uid}.{ext}"
        f.save(os.path.join(UPLOAD_DIR,"avatars",fname))
        url = f"/uploads/avatars/{fname}"
        with get_db() as db:
            db.execute("UPDATE users SET avatar_url=?,avatar_data='' WHERE id=?",(url,uid)); db.commit()
        return ok({"avatar_url":url})
    b64 = request.form.get("base64","")
    if b64:
        with get_db() as db:
            db.execute("UPDATE users SET avatar_data=?,avatar_url='' WHERE id=?",(b64[:500000],uid)); db.commit()
        return ok({"avatar_data":b64[:100]+"..."})
    return err("未提供图片")

@app.put("/api/me/prefs")
@auth_req
def update_prefs():
    d = request.json or {}; uid = request.user["id"]
    for k in ["bg_scene","bg_custom_url","daily_goal_min"]:
        if k in d:
            with get_db() as db:
                db.execute(f"UPDATE user_prefs SET {k}=? WHERE user_id=?",(d[k],uid)); db.commit()
    return ok()

@app.post("/api/me/rank_bg")
@auth_req
def upload_rank_bg():
    uid = request.user["id"]; period = request.args.get("period","daily")
    rank_data = _get_ranking(period)
    if not rank_data or rank_data[0]["user_id"] != uid:
        return err("只有第一名才能设置排行榜背景")
    f = request.files.get("bg"); b64 = request.form.get("base64","")
    if f and f.filename:
        ext = f.filename.rsplit(".",1)[-1].lower()
        fname = f"{uid}_rankbg.{ext}"
        f.save(os.path.join(UPLOAD_DIR,"rank_bgs",fname)); url = f"/uploads/rank_bgs/{fname}"
    elif b64:
        try:
            _, data = b64.split(",",1)
            fname = f"{uid}_rankbg.jpg"
            with open(os.path.join(UPLOAD_DIR,"rank_bgs",fname),"wb") as fp: fp.write(base64.b64decode(data))
            url = f"/uploads/rank_bgs/{fname}"
        except: return err("图片格式错误")
    else: return err("未提供图片")
    with get_db() as db:
        db.execute("UPDATE users SET rank_bg_url=? WHERE id=?",(url,uid)); db.commit()
    return ok({"rank_bg_url":url})

# ══════════════════════════════════════════════════════════════════
# 排名（100条数据）
# ══════════════════════════════════════════════════════════════════
def _get_ranking(period="daily"):
    now = time.time()
    since = {
        "daily":   now - 86400,
        "weekly":  now - 86400*7,
        "monthly": now - 86400*30,
    }.get(period, now-86400)
    with get_db() as db:
        rows = db.execute("""
            SELECT s.user_id, u.username, u.avatar_url, u.avatar_data, u.rank_bg_url,
                   COALESCE(SUM(s.duration_min),0) total_min,
                   COALESCE(AVG(s.avg_focus)*100,0) avg_focus,
                   COALESCE(SUM(s.pomodoros),0) pomodoros
            FROM sessions s JOIN users u ON s.user_id=u.id
            WHERE s.end_time IS NOT NULL AND s.start_time>? AND u.join_ranking=1
            GROUP BY s.user_id ORDER BY total_min DESC LIMIT 100
        """,(since,)).fetchall()
    return [dict(r, rank=i+1, total_min=round(r["total_min"],1), avg_focus=round(r["avg_focus"],1))
            for i,r in enumerate(rows)]

@app.get("/api/ranking/<period>")
def ranking(period):
    if period not in("daily","weekly","monthly"): return err("无效类型")
    return ok(_get_ranking(period))

# ══════════════════════════════════════════════════════════════════
# 学习计划
# ══════════════════════════════════════════════════════════════════
@app.get("/api/plans")
@auth_req
def list_plans():
    with get_db() as db:
        rows = db.execute("SELECT * FROM study_plans WHERE user_id=? ORDER BY sort_order,created_at",
                          (request.user["id"],)).fetchall()
    return ok([dict(r) for r in rows])

@app.post("/api/plans")
@auth_req
def create_plan():
    d = request.json or {}; t = d.get("title","").strip()
    if not t: return err("标题不能为空")
    pid = str(uuid.uuid4())
    with get_db() as db:
        db.execute("INSERT INTO study_plans(id,user_id,title,description,duration_min,subject,color,sort_order) VALUES(?,?,?,?,?,?,?,?)",
            (pid,request.user["id"],t,d.get("description",""),int(d.get("duration_min",25)),
             d.get("subject",""),d.get("color","#c49a3c"),d.get("sort_order",0))); db.commit()
    return ok({"plan_id":pid})

@app.put("/api/plans/<pid>")
@auth_req
def update_plan(pid):
    d = request.json or {}
    with get_db() as db:
        for k in ["title","description","duration_min","subject","color","is_completed"]:
            if k in d: db.execute(f"UPDATE study_plans SET {k}=? WHERE id=? AND user_id=?",(d[k],pid,request.user["id"]))
        db.commit()
    return ok()

@app.delete("/api/plans/<pid>")
@auth_req
def delete_plan(pid):
    with get_db() as db:
        db.execute("DELETE FROM study_plans WHERE id=? AND user_id=?",(pid,request.user["id"])); db.commit()
    return ok()

# ══════════════════════════════════════════════════════════════════
# 学习会话
# ══════════════════════════════════════════════════════════════════
@app.post("/api/sessions/start")
@auth_req
def sess_start():
    d = request.json or {}; sid = str(uuid.uuid4()); now = time.time()
    with get_db() as db:
        db.execute("INSERT INTO sessions(id,user_id,start_time,plan_id,hour_of_day) VALUES(?,?,?,?,?)",
                   (sid,request.user["id"],now,d.get("plan_id"),datetime.now().hour)); db.commit()
    return ok({"session_id":sid})

@app.post("/api/sessions/<sid>/end")
@auth_req
def sess_end(sid):
    d = request.json or {}; now = time.time()
    with get_db() as db:
        s = db.execute("SELECT * FROM sessions WHERE id=? AND user_id=?",(sid,request.user["id"])).fetchone()
        if not s: return err("会话不存在",404)
        af = db.execute("SELECT AVG(focus_score) f FROM focus_log WHERE session_id=?",(sid,)).fetchone()["f"] or 0
        dur = (now - s["start_time"])/60
        db.execute("UPDATE sessions SET end_time=?,duration_min=?,avg_focus=?,pomodoros=? WHERE id=?",
                   (now,round(dur,2),round(af,3),d.get("pomodoros",0),sid))
        rw = db.execute("SELECT * FROM user_rewards WHERE user_id=?",(request.user["id"],)).fetchone()
        new_t = (rw["total_minutes"] if rw else 0)+int(dur)
        unlocked = json.loads(rw["unlocked_skins"] if rw else '["default"]')
        for mins,skin in[(60,"cozy"),(300,"pro"),(1000,"legend"),(3000,"master")]:
            if new_t>=mins and skin not in unlocked: unlocked.append(skin)
        db.execute("UPDATE user_rewards SET total_minutes=?,unlocked_skins=? WHERE user_id=?",
                   (new_t,json.dumps(unlocked),request.user["id"]))
        db.commit()
    # 异步更新学习画像
    _update_learning_profile(request.user["id"])
    return ok({"duration_min":round(dur,2),"avg_focus":round(af*100,1)})

@app.post("/api/sessions/<sid>/focus")
@auth_req
def log_focus(sid):
    d = request.json or {}
    with get_db() as db:
        db.execute("INSERT INTO focus_log(session_id,timestamp,focus_score,fatigue,is_away,source) VALUES(?,?,?,?,?,?)",
                   (sid,time.time(),d.get("focus",0.5),d.get("fatigue",0),
                    int(d.get("is_away",False)),d.get("source","behavior"))); db.commit()
    return ok()

# ── 行为日志接口（鼠标/键盘/空闲事件）
@app.post("/api/sessions/<sid>/behavior")
@auth_req
def log_behavior(sid):
    """
    接收前端行为数据计算专注度
    event_type: mouse_move, click, key_press, idle_start, idle_end,
                tab_blur, tab_focus, scroll, quote_view, share_action
    """
    d = request.json or {}
    events = d.get("events", [])
    if not events:
        return ok()

    uid = request.user["id"]
    now = time.time()

    # 计算行为专注度分数
    focus_contribution = _calc_behavior_focus(events)

    with get_db() as db:
        for evt in events[:50]:  # 限制批量大小
            db.execute("""INSERT INTO behavior_log(user_id,session_id,timestamp,event_type,data,focus_contribution)
                VALUES(?,?,?,?,?,?)""",
                (uid, sid, evt.get("ts",now), evt.get("type","unknown"),
                 json.dumps(evt.get("data",{}))[:500], focus_contribution))
        db.commit()

    return ok({"focus_score": focus_contribution})

def _calc_behavior_focus(events) -> float:
    """
    基于行为事件计算专注度分数（0-1）
    算法：
    - 鼠标活跃 + 点击 → 加分
    - 空闲时间 → 减分
    - 标签切换 → 减分
    - 查看语录/分享 → 轻微减分（允许短暂切换）
    - 持续无操作 → 显著减分
    """
    if not events:
        return 0.5

    score = 0.7  # 基础分
    idle_seconds = 0
    tab_switches = 0
    active_events = 0

    for evt in events:
        t = evt.get("type","")
        data = evt.get("data",{})
        if t in ("mouse_move","scroll"):
            active_events += 1
            score = min(1.0, score + 0.005)
        elif t == "click":
            active_events += 1
            score = min(1.0, score + 0.01)
        elif t == "key_press":
            active_events += 1
            score = min(1.0, score + 0.008)
        elif t == "idle_start":
            idle_seconds += data.get("idle_seconds",30)
            penalty = min(0.3, idle_seconds / 300 * 0.3)
            score = max(0.1, score - penalty)
        elif t == "tab_blur":
            tab_switches += 1
            score = max(0.2, score - 0.08)
        elif t == "tab_focus":
            score = min(score + 0.03, 0.8)  # 回来了，恢复一点
        elif t in ("quote_view","share_action"):
            score = max(0.3, score - 0.02)  # 轻微减分

    # 没有任何活跃事件 → 降分
    if active_events == 0:
        score = max(0.15, score - 0.15)

    return round(min(1.0, max(0.0, score)), 3)

@app.get("/api/sessions/history")
@auth_req
def sess_history():
    with get_db() as db:
        rows = db.execute("""SELECT s.*,p.title plan_title FROM sessions s
            LEFT JOIN study_plans p ON s.plan_id=p.id
            WHERE s.user_id=? AND s.end_time IS NOT NULL
            ORDER BY s.start_time DESC LIMIT 30""",(request.user["id"],)).fetchall()
    return ok([dict(r) for r in rows])

@app.get("/api/stats/monthly")
@auth_req
def monthly_stats():
    """近30天每日学习数据，供前端绘制图表"""
    uid = request.user["id"]
    with get_db() as db:
        rows = db.execute("""
            SELECT DATE(start_time,'unixepoch','localtime') day,
                   COALESCE(SUM(duration_min),0)   total_min,
                   COALESCE(AVG(avg_focus)*100, 0) avg_focus,
                   COALESCE(SUM(pomodoros),0)       pomodoros
            FROM sessions
            WHERE user_id=? AND end_time IS NOT NULL
              AND start_time > unixepoch('now','-30 days')
            GROUP BY day ORDER BY day
        """, (uid,)).fetchall()
        subj_rows = db.execute("""
            SELECT COALESCE(p.subject,'其他') subject,
                   COALESCE(SUM(s.duration_min),0) total_min,
                   COALESCE(AVG(s.avg_focus)*100,0) avg_focus,
                   COUNT(*) sessions
            FROM sessions s
            LEFT JOIN study_plans p ON s.plan_id=p.id
            WHERE s.user_id=? AND s.end_time IS NOT NULL
              AND s.start_time > unixepoch('now','-30 days')
            GROUP BY subject ORDER BY total_min DESC LIMIT 8
        """, (uid,)).fetchall()
    return ok({
        "daily": [dict(r) for r in rows],
        "subjects": [dict(r) for r in subj_rows],
    })

@app.get("/api/stats/weekly")
@auth_req
def weekly_stats():
    uid = request.user["id"]
    with get_db() as db:
        rows = db.execute("""SELECT DATE(start_time,'unixepoch','localtime') day,
            COALESCE(SUM(duration_min),0) total_min,
            COALESCE(AVG(avg_focus)*100,0) avg_focus,
            COALESCE(SUM(pomodoros),0) pomodoros
            FROM sessions WHERE user_id=? AND end_time IS NOT NULL
              AND start_time>unixepoch('now','-7 days')
            GROUP BY day ORDER BY day""",(uid,)).fetchall()
        result = {}
        for i in range(6,-1,-1):
            day = (datetime.now()-timedelta(days=i)).strftime("%Y-%m-%d")
            result[day] = {"day":day,"total_min":0,"avg_focus":0,"pomodoros":0}
        for r in rows:
            if r["day"] in result: result[r["day"]] = dict(r)
        total = db.execute("""SELECT COALESCE(SUM(duration_min),0) m,
            COALESCE(AVG(avg_focus)*100,0) f,COALESCE(SUM(pomodoros),0) p
            FROM sessions WHERE user_id=? AND end_time IS NOT NULL""",(uid,)).fetchone()
    return ok({"daily":list(result.values()),"total":dict(total)})

# ══════════════════════════════════════════════════════════════════
# 个性化学习建模
# ══════════════════════════════════════════════════════════════════
def _update_learning_profile(uid: str):
    """分析用户学习数据，更新学习画像"""
    try:
        with get_db() as db:
            # 获取所有学习会话
            sessions = db.execute("""
                SELECT duration_min, avg_focus, hour_of_day, start_time, end_time
                FROM sessions WHERE user_id=? AND end_time IS NOT NULL
                ORDER BY start_time DESC LIMIT 100
            """, (uid,)).fetchall()

            if not sessions:
                return

            # 按小时统计平均专注度
            hour_focus = {}
            for s in sessions:
                h = s["hour_of_day"] or 0
                if h not in hour_focus: hour_focus[h] = []
                hour_focus[h].append(s["avg_focus"] or 0)

            # 找最佳小时
            best_hour = max(hour_focus.items(), key=lambda x: sum(x[1])/len(x[1]),
                           default=(9, [0.65]))[0] if hour_focus else 9

            # 低效时段（专注度低于均值20%的时段）
            overall_avg = sum(s["avg_focus"] or 0 for s in sessions) / len(sessions)
            weak_hours = [h for h,vals in hour_focus.items()
                         if sum(vals)/len(vals) < overall_avg * 0.8]

            # 平均专注度
            focus_avg = round(overall_avg, 3)

            # 平均单次时长
            avg_session = round(sum(s["duration_min"] or 0 for s in sessions) / len(sessions), 1)

            # 学习偏好（短时多次 vs 长时持续）
            short_sessions = sum(1 for s in sessions if (s["duration_min"] or 0) < 35)
            pref = "short" if short_sessions > len(sessions)*0.6 else "long"

            # 连续打卡天数
            dates = sorted(set(datetime.fromtimestamp(s["start_time"]).date()
                               for s in sessions if s["start_time"]), reverse=True)
            streak = 0
            today = datetime.now().date()
            for i, d in enumerate(dates):
                expected = today - timedelta(days=i)
                if d == expected: streak += 1
                else: break

            last_date = str(dates[0]) if dates else ""

            db.execute("""INSERT OR REPLACE INTO learning_profile
                (user_id,best_hour,avg_session_min,focus_avg,streak_days,
                 last_study_date,total_sessions,weak_hours,pref_session_type,last_updated)
                VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (uid, best_hour, avg_session, focus_avg, streak,
                 last_date, len(sessions), json.dumps(weak_hours[:3]),
                 pref, time.time()))
            db.commit()
    except Exception as e:
        print(f"update_profile error: {e}")

@app.get("/api/profile/analysis")
@auth_req
def get_profile_analysis():
    """获取完整学习画像分析"""
    uid = request.user["id"]
    _update_learning_profile(uid)

    with get_db() as db:
        lp = db.execute("SELECT * FROM learning_profile WHERE user_id=?",(uid,)).fetchone()
        # 按小时分布
        hour_data = db.execute("""
            SELECT hour_of_day h, AVG(avg_focus)*100 focus, SUM(duration_min) total_min, COUNT(*) cnt
            FROM sessions WHERE user_id=? AND end_time IS NOT NULL
            GROUP BY hour_of_day ORDER BY h
        """,(uid,)).fetchall()
        # 按学科分布
        subject_data = db.execute("""
            SELECT p.subject, SUM(s.duration_min) total_min, AVG(s.avg_focus)*100 avg_focus
            FROM sessions s JOIN study_plans p ON s.plan_id=p.id
            WHERE s.user_id=? AND s.end_time IS NOT NULL AND p.subject!=''
            GROUP BY p.subject ORDER BY total_min DESC LIMIT 8
        """,(uid,)).fetchall()
        # 最近30天趋势
        trend = db.execute("""
            SELECT DATE(start_time,'unixepoch','localtime') day,
                   SUM(duration_min) total_min, AVG(avg_focus)*100 avg_focus
            FROM sessions WHERE user_id=? AND end_time IS NOT NULL
              AND start_time>unixepoch('now','-30 days')
            GROUP BY day ORDER BY day
        """,(uid,)).fetchall()

    profile_data = dict(lp) if lp else {}
    weak_hours = json.loads(profile_data.get("weak_hours","[]"))

    # 生成AI建议
    advice = generate_learning_advice(profile_data, request.user["username"])

    return ok({
        "profile": profile_data,
        "hour_distribution": [dict(r) for r in hour_data],
        "subject_distribution": [dict(r) for r in subject_data],
        "trend_30days": [dict(r) for r in trend],
        "ai_advice": advice,
        "summary": {
            "best_hour": profile_data.get("best_hour", 9),
            "best_time_label": f"{profile_data.get('best_hour',9):02d}:00 - {(profile_data.get('best_hour',9)+2) % 24:02d}:00",
            "streak": profile_data.get("streak_days",0),
            "pref_type": "短时高频" if profile_data.get("pref_session_type")=="short" else "长时深入",
            "weak_hours_label": [f"{h}点" for h in weak_hours],
        }
    })

@app.get("/api/profile/recommend_plan")
@auth_req
def recommend_plan():
    """基于学习画像推荐个性化学习计划"""
    uid = request.user["id"]
    with get_db() as db:
        lp = db.execute("SELECT * FROM learning_profile WHERE user_id=?",(uid,)).fetchone()
        existing = db.execute("SELECT title,duration_min,subject FROM study_plans WHERE user_id=? AND is_completed=0",
                              (uid,)).fetchall()

    profile = dict(lp) if lp else {}
    pref = profile.get("pref_session_type","short")
    best_h = profile.get("best_hour",9)
    avg_s = profile.get("avg_session_min",45)

    existing_titles = [r["title"] for r in existing]

    prompt = (
        f"你是学习顾问，为用户推荐学习计划安排。"
        f"用户偏好：{pref=='short' and '短时多次（番茄钟）' or '长时深入'}，"
        f"最佳时段：{best_h}点，平均每次{avg_s:.0f}分钟。"
        f"已有计划：{','.join(existing_titles[:3]) or '暂无'}。"
        f"推荐今天的3个具体学习安排（格式：任务名|时长分钟|建议时间，用换行分隔），只输出这3行。"
    )
    result = groq_chat([{"role":"user","content":prompt}], max_tokens=150, temperature=0.7)

    # 解析建议
    recommendations = []
    if result:
        for line in result.strip().split("\n"):
            parts = line.split("|")
            if len(parts) >= 2:
                try:
                    recommendations.append({
                        "title": parts[0].strip().lstrip("123.、）").strip(),
                        "duration_min": int(''.join(filter(str.isdigit, parts[1])) or 25),
                        "suggested_time": parts[2].strip() if len(parts)>2 else f"{best_h}:00",
                    })
                except: pass

    if not recommendations:
        dur = 25 if pref=="short" else 50
        recommendations = [
            {"title":"专注学习A","duration_min":dur,"suggested_time":f"{best_h}:00"},
            {"title":"专注学习B","duration_min":dur,"suggested_time":f"{(best_h+2)%24}:00"},
        ]

    return ok({"recommendations":recommendations,"based_on":profile.get("pref_session_type","short")})

# ══════════════════════════════════════════════════════════════════
# 打卡 / 语录 / 社区
# ══════════════════════════════════════════════════════════════════
@app.post("/api/checkins")
@auth_req
def checkin():
    note=request.form.get("note",""); sid=request.form.get("session_id")
    pub=int(request.form.get("is_public",1)); path=""
    f=request.files.get("photo")
    if f and f.filename:
        ext=f.filename.rsplit(".",1)[-1].lower()
        if ext not in("jpg","jpeg","png","webp"): return err("只支持jpg/png/webp")
        fname=f"{uuid.uuid4()}.{ext}"; f.save(os.path.join(UPLOAD_DIR,fname)); path=f"/uploads/{fname}"
    cid=str(uuid.uuid4())
    with get_db() as db:
        db.execute("INSERT INTO checkins(id,user_id,session_id,photo_path,note,is_public) VALUES(?,?,?,?,?,?)",
                   (cid,request.user["id"],sid,path,note,pub)); db.commit()
    return ok({"checkin_id":cid,"photo_url":path})

@app.get("/api/quotes/random")
def rnd_quote():
    cat=request.args.get("category")
    with get_db() as db:
        q="SELECT * FROM quotes WHERE category=? ORDER BY RANDOM() LIMIT 1" if cat else "SELECT * FROM quotes ORDER BY RANDOM() LIMIT 1"
        r=db.execute(q,(cat,) if cat else ()).fetchone()
    return ok(dict(r) if r else {"text":"继续加油！","author":"","id":0})

@app.get("/api/quotes")
def list_quotes():
    with get_db() as db:
        rows=db.execute("SELECT * FROM quotes ORDER BY id DESC LIMIT 200").fetchall()
    return ok([dict(r) for r in rows])

@app.post("/api/quotes")
@auth_req
def add_quote():
    d=request.json or {}; t=d.get("text","").strip()
    if not t: return err("内容不能为空")
    with get_db() as db:
        db.execute("INSERT INTO quotes(text,author,category,user_id,is_custom) VALUES(?,?,?,?,1)",
                   (t,d.get("author",""),d.get("category","general"),request.user["id"])); db.commit()
    return ok()

@app.get("/api/posts")
def list_posts():
    with get_db() as db:
        rows=db.execute("""SELECT p.*,u.username,u.avatar_url,u.avatar_data,
            (SELECT COUNT(*) FROM comments WHERE post_id=p.id) comment_count
            FROM posts p JOIN users u ON p.user_id=u.id
            ORDER BY p.created_at DESC LIMIT 50""").fetchall()
    return ok([dict(r) for r in rows])

@app.post("/api/posts")
@auth_req
def new_post():
    content=request.form.get("content","").strip() or (request.json or {}).get("content","").strip()
    if not content: return err("内容不能为空")
    image_url=""
    f=request.files.get("image")
    if f and f.filename:
        ext=f.filename.rsplit(".",1)[-1].lower()
        if ext in("jpg","jpeg","png","webp","gif"):
            fname=f"{uuid.uuid4()}.{ext}"; f.save(os.path.join(UPLOAD_DIR,"posts",fname)); image_url=f"/uploads/posts/{fname}"
    pid=str(uuid.uuid4())
    with get_db() as db:
        db.execute("INSERT INTO posts(id,user_id,content,image_url) VALUES(?,?,?,?)",
                   (pid,request.user["id"],content,image_url)); db.commit()
    return ok({"post_id":pid})

@app.post("/api/posts/<pid>/like")
@auth_req
def like_post(pid):
    uid=request.user["id"]
    with get_db() as db:
        ex=db.execute("SELECT 1 FROM post_likes WHERE post_id=? AND user_id=?",(pid,uid)).fetchone()
        if ex:
            db.execute("DELETE FROM post_likes WHERE post_id=? AND user_id=?",(pid,uid))
            db.execute("UPDATE posts SET likes=MAX(0,likes-1) WHERE id=?",(pid,)); liked=False
        else:
            db.execute("INSERT INTO post_likes VALUES(?,?)",(pid,uid))
            db.execute("UPDATE posts SET likes=likes+1 WHERE id=?",(pid,)); liked=True
        db.commit()
        lk=db.execute("SELECT likes FROM posts WHERE id=?",(pid,)).fetchone()["likes"]
    return ok({"liked":liked,"likes":lk})

@app.post("/api/posts/<pid>/comments")
@auth_req
def add_comment(pid):
    d=request.json or {}; c=d.get("content","").strip()
    if not c: return err("评论不能为空")
    with get_db() as db:
        db.execute("INSERT INTO comments(post_id,user_id,content) VALUES(?,?,?)",
                   (pid,request.user["id"],c)); db.commit()
    return ok()

@app.get("/api/posts/<pid>/comments")
def get_comments(pid):
    with get_db() as db:
        rows=db.execute("""SELECT c.*,u.username,u.avatar_url,u.avatar_data FROM comments c
            JOIN users u ON c.user_id=u.id WHERE c.post_id=? ORDER BY c.created_at""",(pid,)).fetchall()
    return ok([dict(r) for r in rows])

# ══════════════════════════════════════════════════════════════════
# AI 对话（保存历史，有上下文）
# ══════════════════════════════════════════════════════════════════
@app.post("/api/ai/chat")
@auth_req
def ai_chat():
    d = request.json or {}
    msg = d.get("message","").strip()
    ctx = d.get("context","")
    if not msg: return err("消息不能为空")
    uid = request.user["id"]

    # 获取最近对话历史（保证上下文连贯）
    with get_db() as db:
        hist = db.execute("""SELECT role,content FROM chat_history
            WHERE user_id=? ORDER BY created_at DESC LIMIT 8""",(uid,)).fetchall()

    history = [{"role":h["role"],"content":h["content"]} for h in reversed(hist)]
    result = lamp_reply(msg, ctx, history)
    fallbacks = ["这个问题很有意思，继续想一想！","你的思路对，坚持下去！",
                 "遇到困难很正常，分解一下再试试。","休息一下，思路会更清晰。"]
    reply = result or random.choice(fallbacks)

    # 保存对话历史
    with get_db() as db:
        db.execute("INSERT INTO chat_history(user_id,role,content) VALUES(?,?,?)",(uid,"user",msg))
        db.execute("INSERT INTO chat_history(user_id,role,content) VALUES(?,?,?)",(uid,"assistant",reply))
        db.execute("""DELETE FROM chat_history WHERE user_id=? AND id NOT IN
            (SELECT id FROM chat_history WHERE user_id=? ORDER BY created_at DESC LIMIT 50)""",(uid,uid))
        db.commit()

    return ok({"reply":reply,"ai_powered":bool(result)})

@app.post("/api/ai/remind")
@auth_req
def ai_remind_api():
    d = request.json or {}
    plan_title = d.get("plan_title","学习任务")
    elapsed = float(d.get("elapsed_min",0))
    total_today = float(d.get("total_today_min",0))
    msg = generate_remind_msg(request.user["username"], plan_title, elapsed, total_today)
    return ok({"message":msg})

@app.post("/api/decision")
def decision():
    d = request.json or {}
    foc = float(d.get("focus",0.5)); dur = float(d.get("duration_min",0))
    fat = float(d.get("fatigue",0)); away = bool(d.get("is_away",False))
    asec = float(d.get("away_sec",0))

    if away and asec>120:    act,lv = "alert_away","warning"
    elif dur>=90:             act,lv = "force_rest","danger"
    elif fat>=0.75:           act,lv = "suggest_rest","warning"
    elif foc<0.35 and dur>2:  act,lv = "remind_focus","warning"
    elif foc>=0.85 and dur>1: act,lv = "praise","success"
    else:                     act,lv = "encourage","info"

    msg = generate_decision_msg(act, foc, dur)
    return ok({"action":act,"message":msg,"level":lv,"ai_powered":bool(get_groq())})

@app.get("/api/health")
def health():
    g = get_groq()
    return ok({"status":"running","ai_enabled":bool(GROQ_KEY),
               "ai_connected":bool(g),
               "model": getattr(groq_chat, '_working_model', 'llama-3.3-70b-versatile'),
               "version":"6.0.0"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    debug = os.environ.get("FLASK_ENV","development") == "development"
    print(f"\n🚀  http://localhost:{port}")
    g = get_groq()
    print(f"🤖  AI: {'✅ Groq 已连接（自动选择可用模型）' if g else '⚠️  AI未连接（检查GROQ_API_KEY）'}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
