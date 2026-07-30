const express = require('express');
const cors = require('cors');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 5000;

app.use(express.json({ limit: '5mb' }));
app.use(cors());
app.use(express.static(require('path').join(__dirname, 'public')));

let usersDB = [
  { username: "student", password: "password123", name: "Demo Student", grade: "Grade 11", curriculum: "AP STEM", classList: "Calculus, AP Biology, World History", bio: "Targeting top engineering schools.", profilePic: null, commonClasses: [], likedReels: [], friends: [], friendRequests: [], sentRequests: [], isDemo: true },
  { username: "maya", password: "password123", name: "Maya Chen", grade: "Grade 11", curriculum: "AP STEM", classList: "Calculus, AP Chemistry, World History", bio: "Physics olympiad prep.", profilePic: null, commonClasses: [], likedReels: [], friends: [], friendRequests: [], sentRequests: [], isDemo: true },
  { username: "jordan", password: "password123", name: "Jordan Lee", grade: "Grade 10", curriculum: "IB", classList: "IB Biology, IB Math", bio: "Bio nerd, always down to quiz.", profilePic: null, commonClasses: [], likedReels: [], friends: [], friendRequests: [], sentRequests: [], isDemo: true }
];

// sharedReelsDB[username] = array of { reel, fromUsername, ts } — reels sent TO this user
let sharedReelsDB = {};

let workspaceNotesDB = [
  { id: 101, username: "student", title: "Calculus Limits & Derivatives", content: "Derivatives represent the instantaneous slope of a tangent line to a curve." }
];

let cachedReelsDB = [];

// ── PERSONALIZED FEED RECOMMENDER ──────────────────────────────────────────
// Weights trained in Colab (see ml/studyreels_recommender_training.ipynb) on
// simulated interaction logs — a small logistic regression predicting
// P(engagement) for a (student, reel) pair. Re-run the notebook on real
// logged activity once you have enough of it, then swap these numbers.
const RECOMMENDER_MODEL = {
  weights: {
    subj_hist_rate: 1.3358791404406183,
    type_hist_rate: 0.313565225082944,
    overall_engage_rate: -0.16770611302291547,
    subj_view_count_norm: 0.023441641383748286,
    novelty: 0.4179239235466139
  },
  intercept: -0.9376634973778843,
  features: ["subj_hist_rate", "type_hist_rate", "overall_engage_rate", "subj_view_count_norm", "novelty"]
};

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function scoreReel(features) {
  let z = RECOMMENDER_MODEL.intercept;
  for (const f of RECOMMENDER_MODEL.features) {
    z += RECOMMENDER_MODEL.weights[f] * (features[f] || 0);
  }
  return sigmoid(z);
}

// activityDB[username] = array of { reelId, subject, type, watchSeconds, liked, quizCorrect, ts }
let activityDB = {};

function getUserHistory(username) {
  return activityDB[username] || [];
}

// Builds the same 5 causal features the model was trained on, from a user's
// logged history up to now, for one candidate reel.
function buildFeatures(history, reel) {
  const subject = (reel.semanticTags && reel.semanticTags[0]) || "STEM";
  const subjEvents = history.filter(h => h.subject === subject);
  const typeEvents = history.filter(h => h.type === reel.type);
  const seenThisReel = history.some(h => h.reelId === reel.id);

  const rate = (events) => events.length ? events.filter(e => e.engaged).length / events.length : 0.5;

  return {
    subj_hist_rate: rate(subjEvents),
    type_hist_rate: rate(typeEvents),
    overall_engage_rate: rate(history),
    subj_view_count_norm: Math.min(Math.log1p(subjEvents.length) / Math.log1p(15), 1.0),
    novelty: seenThisReel ? 0 : 1
  };
}

// 1. AUTH ROUTES
app.post('/api/auth/signup', (req, res) => {
  const { username, password, name, grade, curriculum, classList, bio, profilePic, commonClasses } = req.body;
  if (!username || !password || !name) {
    return res.status(400).json({ error: "Missing required fields." });
  }
  const existing = usersDB.find(u => u.username === username);
  if (existing) {
    return res.status(400).json({ error: "Username already taken." });
  }
  const newUser = {
    username, password, name,
    grade: grade || "", curriculum: curriculum || "",
    classList: classList || "", bio: bio || "",
    profilePic: profilePic || null,
    commonClasses: Array.isArray(commonClasses) ? commonClasses : [],
    likedReels: [], friends: [], friendRequests: [], sentRequests: []
  };
  usersDB.push(newUser);
  res.json({ success: true, user: newUser });
});

app.post('/api/auth/signin', (req, res) => {
  const { username, password } = req.body;
  const user = usersDB.find(u => u.username === username && u.password === password);
  if (user) {
    res.json({ success: true, user });
  } else {
    res.status(401).json({ error: "Invalid username or password." });
  }
});

// 2. PROFILE UPDATE ROUTE
app.post('/api/user/update', (req, res) => {
  const { username, name, grade, curriculum, classList, bio, profilePic, commonClasses } = req.body;
  const user = usersDB.find(u => u.username === username);
  if (user) {
    user.name = name || user.name;
    user.grade = grade || user.grade;
    user.curriculum = curriculum || user.curriculum;
    user.classList = classList !== undefined ? classList : user.classList;
    user.bio = bio !== undefined ? bio : user.bio;
    user.profilePic = profilePic !== undefined ? profilePic : user.profilePic;
    user.commonClasses = Array.isArray(commonClasses) ? commonClasses : (user.commonClasses || []);
    res.json({ success: true, user });
  } else {
    res.status(404).json({ error: "User not found" });
  }
});

// 3. LIKED REELS ROUTE
app.post('/api/user/like', (req, res) => {
  const { username, reel } = req.body;
  const user = usersDB.find(u => u.username === username);
  if (user) {
    const exists = user.likedReels.some(r => r.id === reel.id);
    if (!exists) {
      user.likedReels.push(reel);
    } else {
      user.likedReels = user.likedReels.filter(r => r.id !== reel.id); // Toggle off if already liked
    }
    res.json({ success: true, likedReels: user.likedReels });
  } else {
    res.status(404).json({ error: "User not found" });
  }
});

// Reel type presets — must match the gradient classes/keys the frontend expects (bgGradient, not backgroundStyle)
const REEL_STYLES = [
  { type: "pov", bgGradient: "bg-gradient-to-tr from-rose-500 via-pink-500 to-amber-400 text-white shadow-xl" },
  { type: "ai_footage", bgGradient: "bg-gradient-to-br from-emerald-600 via-teal-700 to-slate-900 text-white shadow-xl" },
  { type: "avatar", bgGradient: "bg-gradient-to-tr from-indigo-600 via-blue-600 to-slate-900 text-white shadow-xl" }
];

function guessTags(topic) {
  const t = topic.toLowerCase();
  if (/calc|algebra|math|geometry|trig/.test(t)) return ["Math", "STEM"];
  if (/bio|cell|chem|physic|photosynth/.test(t)) return ["Biology", "STEM"];
  if (/hist|war|econ|govern|literat/.test(t)) return ["Humanities"];
  return ["STEM"];
}

// ── TRUST SCORE ENGINE ──────────────────────────────────────────────────
// Deterministic, rule-based grounding check — NOT another AI model grading
// the first one. Every number in the score traces back to an inspectable
// rule, not to a second model's opinion (which would just be another black box).
const STOPWORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","is","are",
  "was","were","be","been","this","that","these","those","it","as","by","from","has",
  "have","had","will","can","could","would","should","not","no","so","if","than","then",
  "into","about","over","under","more","most","such","also","when","while","which","who",
  "whom","its","their","they","we","you","your","our","i","he","she","them","his","her",
  "what","how","why","each","between","because","just","like","get","gets","one","two"
]);

function tokenize(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(w => w.length > 3 && !STOPWORDS.has(w));
}

function computeTrustScore(generatedText, sourceText, quiz) {
  const sourceTerms = new Set(tokenize(sourceText));
  const genTermsUnique = [...new Set(tokenize(generatedText))];

  const matchedTerms = genTermsUnique.filter(t => sourceTerms.has(t));
  const unmatchedTerms = genTermsUnique.filter(t => !sourceTerms.has(t));
  const termOverlapPct = genTermsUnique.length
    ? Math.round((matchedTerms.length / genTermsUnique.length) * 100)
    : 100;

  const genNumbers = generatedText.match(/\b\d+(\.\d+)?\b/g) || [];
  const sourceNumbers = new Set(sourceText.match(/\b\d+(\.\d+)?\b/g) || []);
  const unverifiedNumbers = [...new Set(genNumbers.filter(n => !sourceNumbers.has(n)))];

  const correctAnswerText = (quiz && quiz.options && quiz.options[quiz.correctIndex]) || "";
  const answerTerms = tokenize(correctAnswerText);
  const quizAnswerVerified = answerTerms.length === 0 ? true : answerTerms.some(t => sourceTerms.has(t));

  let score = termOverlapPct * 0.6;
  score += quizAnswerVerified ? 25 : 0;
  score += unverifiedNumbers.length === 0 ? 15 : Math.max(0, 15 - unverifiedNumbers.length * 7);
  score = Math.round(Math.min(100, Math.max(0, score)));

  const flags = [];
  if (unverifiedNumbers.length > 0) {
    flags.push(`${unverifiedNumbers.length} number(s) appear in the reel but not in your notes: ${unverifiedNumbers.slice(0, 4).join(", ")}`);
  }
  if (!quizAnswerVerified) {
    flags.push("The quiz's correct answer isn't clearly traceable to your source notes.");
  }
  if (termOverlapPct < 50) {
    flags.push("Less than half of this reel's key terms come from your uploaded notes.");
  }

  return { score, termOverlapPct, matchedTerms: matchedTerms.slice(0, 12), unmatchedTerms: unmatchedTerms.slice(0, 8), unverifiedNumbers, quizAnswerVerified, flags };
}

// 4. AI GENERATION & SMART QUIZ ROUTE
app.post('/api/ai/generate-reel', async (req, res) => {
  const { topic, content } = req.body;
  if (!topic || !content) {
    return res.status(400).json({ error: "Both topic and content are required." });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  const style = REEL_STYLES[Math.floor(Math.random() * REEL_STYLES.length)];

  // Local fallback (used only if no key is configured or the API call fails)
  let script = `POV: when your exam is tomorrow and you only just uploaded notes on ${topic}.`;
  let description = content.length > 280 ? content.slice(0, 277) + "..." : content;
  let quiz = {
    question: `Based on your notes, what's the core idea behind ${topic}?`,
    options: [
      `The primary mechanism described in your notes`,
      "An unrelated random historical fact",
      "A generic computational error",
      "Completely unverified external trivia"
    ],
    correctIndex: 0
  };
  let usedAI = false;

  if (apiKey && !apiKey.includes("YOUR_ACTUAL_CLAUDE")) {
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01"
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 600,
          messages: [{
            role: "user",
            content: `You are converting a student's study notes into a short-form "study reel" for an app. Here are the notes on "${topic}":\n\n"""${content}"""\n\nRespond with ONLY valid JSON (no markdown fences, no preamble), matching exactly this shape:\n{"script": "a punchy 1-3 sentence Gen-Z POV-style study reel script grounded in the notes above, under 240 characters", "description": "a clear, plain-language explanation of the underlying concept, 2-4 sentences, written for a student who wants to actually understand it (not just the punchy hook) — no slang, grounded strictly in the notes provided", "quiz": {"question": "a specific active-recall question about the notes", "options": ["correct answer first", "plausible wrong answer", "plausible wrong answer", "plausible wrong answer"], "correctIndex": 0}}`
          }]
        })
      });

      if (!response.ok) {
        const errBody = await response.text();
        throw new Error(`Anthropic API returned ${response.status}: ${errBody}`);
      }

      const data = await response.json();
      const rawText = data.content?.[0]?.text || "";
      const cleaned = rawText.replace(/```json|```/g, "").trim();
      const parsed = JSON.parse(cleaned);

      if (parsed.script && parsed.description && parsed.quiz?.question && Array.isArray(parsed.quiz.options)) {
        script = parsed.script;
        description = parsed.description;
        quiz = parsed.quiz;
        usedAI = true;
      }
    } catch (err) {
      console.error("Claude API error (falling back to local template):", err.message);
    }
  }

  const fullGeneratedText = `${script} ${description} ${quiz.question} ${(quiz.options || []).join(" ")}`;
  const trust = computeTrustScore(fullGeneratedText, content, quiz);

  const newReel = {
    id: Date.now(),
    type: style.type,
    topic,
    script,
    description,
    sourceQuote: content,
    bgGradient: style.bgGradient,
    quiz,
    semanticTags: guessTags(topic),
    aiGenerated: usedAI,
    trust
  };

  cachedReelsDB.push(newReel);
  res.json({ success: true, reel: newReel, aiGenerated: usedAI });
});

// 5. ACTIVITY LOGGING — feeds the recommender's rolling history features
app.post('/api/activity/log', (req, res) => {
  const { username, reelId, subject, type, watchSeconds, liked, quizCorrect } = req.body;
  if (!username || reelId === undefined) {
    return res.status(400).json({ error: "username and reelId are required." });
  }

  const engaged = (watchSeconds >= 6) || !!liked || !!quizCorrect;

  if (!activityDB[username]) activityDB[username] = [];
  activityDB[username].push({
    reelId, subject: subject || "STEM", type: type || "pov",
    watchSeconds: watchSeconds || 0, liked: !!liked, quizCorrect: !!quizCorrect,
    engaged, ts: Date.now()
  });

  res.json({ success: true, historyLength: activityDB[username].length });
});

// 6. PERSONALIZED FEED — ranks candidate reels for a user using the trained model
app.post('/api/ai/recommend-feed', (req, res) => {
  const { username, reels } = req.body;
  if (!Array.isArray(reels)) {
    return res.status(400).json({ error: "reels must be an array of candidate reel objects." });
  }

  const history = getUserHistory(username);
  const ranked = reels
    .map(reel => {
      const features = buildFeatures(history, reel);
      return { reel, score: scoreReel(features) };
    })
    .sort((a, b) => b.score - a.score)
    .map(r => ({ ...r.reel, recommendScore: Math.round(r.score * 1000) / 1000 }));

  res.json({ success: true, reels: ranked, historyLength: history.length, personalized: history.length > 0 });
});

// ── FRIENDS / STUDY BUDDY HELPERS ──────────────────────────────────────────
function ensureFriendFields(user) {
  user.friends = user.friends || [];
  user.friendRequests = user.friendRequests || []; // incoming, array of usernames
  user.sentRequests = user.sentRequests || []; // outgoing, array of usernames
  return user;
}

function publicUser(user) {
  const { password, ...safe } = user;
  return safe;
}

// crude overlap score between two comma-separated class/curriculum strings
function overlapScore(a, b) {
  const norm = (s) => (s || "").toLowerCase().split(/[,\/]/).map(x => x.trim()).filter(Boolean);
  const setA = new Set(norm(a));
  const setB = new Set(norm(b));
  let score = 0;
  for (const item of setA) if (setB.has(item)) score++;
  return score;
}

// 7. STUDY BUDDIES & FRIENDS
app.post('/api/users/search', (req, res) => {
  const { username, query } = req.body;
  if (!query) return res.json({ success: true, users: [] });
  const me = usersDB.find(u => u.username === username);
  if (me) ensureFriendFields(me);
  const q = query.toLowerCase();
  const results = usersDB
    .filter(u => u.username !== username)
    .filter(u => u.username.toLowerCase().includes(q) || (u.name || "").toLowerCase().includes(q))
    .map(u => {
      ensureFriendFields(u);
      return {
        ...publicUser(u),
        isFriend: me ? me.friends.includes(u.username) : false,
        requestSent: me ? me.sentRequests.includes(u.username) : false,
        requestReceived: me ? me.friendRequests.includes(u.username) : false
      };
    });
  res.json({ success: true, users: results });
});

app.post('/api/friends/suggest', (req, res) => {
  const { username } = req.body;
  const me = usersDB.find(u => u.username === username);
  if (!me) return res.status(404).json({ error: "User not found" });
  ensureFriendFields(me);

  const suggestions = usersDB
    .filter(u => u.username !== username)
    .filter(u => !me.friends.includes(u.username) && !me.sentRequests.includes(u.username) && !me.friendRequests.includes(u.username))
    .map(u => {
      ensureFriendFields(u);
      const score = overlapScore(me.classList, u.classList) * 2 + overlapScore(me.curriculum, u.curriculum);
      return { ...publicUser(u), matchScore: score };
    })
    .sort((a, b) => b.matchScore - a.matchScore)
    .slice(0, 10);

  res.json({ success: true, suggestions });
});

app.post('/api/friends/request', (req, res) => {
  const { fromUsername, toUsername } = req.body;
  const from = usersDB.find(u => u.username === fromUsername);
  const to = usersDB.find(u => u.username === toUsername);
  if (!from || !to) return res.status(404).json({ error: "User not found" });
  ensureFriendFields(from); ensureFriendFields(to);

  if (from.friends.includes(toUsername)) {
    return res.json({ success: true, alreadyFriends: true });
  }

  if (to.isDemo) {
    // Demo/seed accounts auto-accept instantly so you can demo sending reels without waiting on a second browser
    if (!from.friends.includes(toUsername)) from.friends.push(toUsername);
    if (!to.friends.includes(fromUsername)) to.friends.push(fromUsername);
    from.sentRequests = from.sentRequests.filter(u => u !== toUsername);
    to.friendRequests = to.friendRequests.filter(u => u !== fromUsername);
    return res.json({ success: true, autoAccepted: true });
  }

  if (from.sentRequests.includes(toUsername)) {
    return res.json({ success: true, alreadyRequested: true });
  }
  from.sentRequests.push(toUsername);
  to.friendRequests.push(fromUsername);
  res.json({ success: true });
});

app.post('/api/friends/respond', (req, res) => {
  const { username, fromUsername, accept } = req.body;
  const me = usersDB.find(u => u.username === username);
  const other = usersDB.find(u => u.username === fromUsername);
  if (!me || !other) return res.status(404).json({ error: "User not found" });
  ensureFriendFields(me); ensureFriendFields(other);

  me.friendRequests = me.friendRequests.filter(u => u !== fromUsername);
  other.sentRequests = other.sentRequests.filter(u => u !== username);

  if (accept) {
    if (!me.friends.includes(fromUsername)) me.friends.push(fromUsername);
    if (!other.friends.includes(username)) other.friends.push(username);
  }
  res.json({ success: true });
});

app.post('/api/friends/list', (req, res) => {
  const { username } = req.body;
  const me = usersDB.find(u => u.username === username);
  if (!me) return res.status(404).json({ error: "User not found" });
  ensureFriendFields(me);

  const toPublic = (uname) => {
    const u = usersDB.find(x => x.username === uname);
    return u ? publicUser(u) : { username: uname, name: uname };
  };

  res.json({
    success: true,
    friends: me.friends.map(toPublic),
    incomingRequests: me.friendRequests.map(toPublic),
    sentRequests: me.sentRequests.map(toPublic)
  });
});

// 8. REEL SHARING (send a study reel to a friend, like sending an Instagram post)
app.post('/api/reels/share', (req, res) => {
  const { fromUsername, toUsername, reel } = req.body;
  const from = usersDB.find(u => u.username === fromUsername);
  const to = usersDB.find(u => u.username === toUsername);
  if (!from || !to || !reel) return res.status(400).json({ error: "fromUsername, toUsername, and reel are required." });
  ensureFriendFields(from);

  if (!from.friends.includes(toUsername)) {
    return res.status(403).json({ error: "You can only send reels to friends." });
  }

  if (!sharedReelsDB[toUsername]) sharedReelsDB[toUsername] = [];
  sharedReelsDB[toUsername].unshift({ reel, fromUsername, fromName: from.name, ts: Date.now() });
  res.json({ success: true });
});

app.post('/api/reels/inbox', (req, res) => {
  const { username } = req.body;
  res.json({ success: true, inbox: sharedReelsDB[username] || [] });
});

// 9. AI STUDY BUDDY — an always-available AI chat "friend", no request/accept needed
app.post('/api/ai/study-buddy-chat', async (req, res) => {
  const { history } = req.body;
  if (!Array.isArray(history) || history.length === 0) {
    return res.status(400).json({ error: "history (with at least one message) is required." });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  let reply = "I'm here to help you study! (No Claude API key is configured yet, so I can only give this canned reply — add ANTHROPIC_API_KEY to your .env to unlock real answers.)";
  let usedAI = false;

  if (apiKey && !apiKey.includes("YOUR_ACTUAL_CLAUDE")) {
    try {
      const turns = history.slice(-12).map(m => ({ role: m.role, content: m.content }));
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
          "anthropic-version": "2023-06-01"
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 500,
          system: "You are Study Buddy AI, a friendly, encouraging study partner inside the StudyReels app for a high school student. Explain concepts clearly and simply, offer a quick quiz question when it helps reinforce the idea, and keep answers concise — a few sentences to a short paragraph unless asked for more detail.",
          messages: turns
        })
      });

      if (!response.ok) {
        const errBody = await response.text();
        throw new Error(`Anthropic API returned ${response.status}: ${errBody}`);
      }

      const data = await response.json();
      reply = data.content?.[0]?.text || reply;
      usedAI = true;
    } catch (err) {
      console.error("Study Buddy chat error (falling back to canned reply):", err.message);
    }
  }

  res.json({ success: true, reply, usedAI });
});

app.listen(PORT, () => {
  console.log(`StudyReels AI Pro Backend running live on port ${PORT}`);
});