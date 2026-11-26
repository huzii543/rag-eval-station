/* ============================================================
   app.js —— 检索与评测的全部行为
   原则：
     1. 语义分是离线实算好的真值（DATA.presets[].scores），页面只做排序与展示，
        不现场估计算法；
     2. 字面分在浏览器内按参考实现的公式实时计算（IDF + 词频饱和 + 宽度惩罚）；
     3. 自由输入没有离线语义分，必须显式标为降级通道，绝不伪装成语义检索结果；
     4. 本地存储只承载本会话操作记录，任何读写失败都不影响页面可用。
   ============================================================ */
(function () {
  'use strict';

  var DATA = window.RAG_DATA;
  var $ = function (id) { return document.getElementById(id); };

  if (!DATA || !DATA.corpus || !DATA.presets) {
    document.body.insertAdjacentHTML('afterbegin',
      '<div style="padding:24px;font:14px system-ui">数据文件 data.js 未加载，页面无法运行。</div>');
    return;
  }

  var P = DATA.params;
  var TEXTS = DATA.corpus.texts;
  var SRC = DATA.corpus.src;
  var COUNT = DATA.corpus.chunkCount;
  var N = TEXTS.length;
  var STOP = new Set(DATA.stopwords);
  var LOWER = TEXTS.map(function (t) { return t.toLowerCase(); });
  var PRESETS = DATA.presets;
  var TOPK = P.TOP_K;
  var A = DATA.attack || null;      // 攻击演示数据（缺省时开关自动禁用）
  var CT = DATA.chunkTest || null;  // 「换我的语料」的分块自检样本

  /* ---------- markdown 清洗 / 分块：与参考实现同一套规则 ----------
     clean() = shared/text_clean.py 的 13 步有序替换；
     chunk() = obsidian_rag.chunk()：空行与 # 号断块，只有超长块才退回滑窗。
     只服务于「换我的语料」，打包时用样本逐块比对过（见 chunkSelfTest）。
     注意 Python 的 len() 按码点计数，JS 的 .length 按 UTF-16 单元，
     所以长度判断与滑窗切片都要先 Array.from 拆成码点数组。 */
  function jsClean(text) {
    var steps = [
      [/```[\s\S]*?```/g, ' '],
      [/`([^`]*)`/g, '$1'],
      [/!\[[^\]]*\]\([^)]*\)/g, ' '],
      [/\[([^\]]*)\]\([^)]*\)/g, '$1'],
      [/https?:\/\/\S+/g, ' '],
      [/<[^>]+>/g, ' '],
      [/[\u{1F000}-\u{1FAFF}☀-➿⬀-⯿←-⇀︀-️]/gu, ''],   // emoji / 箭头 / 变体选择符
      [/\[!\w+\]/g, ' '],
      [/[*`>|]/g, ' '],
      [/[ \t]+/g, ' '],
      [/ *\n */g, '\n'],
      [/\n{3,}/g, '\n\n']
    ];
    for (var i = 0; i < steps.length; i++) text = text.replace(steps[i][0], steps[i][1]);
    return text.trim();
  }
  function cps(s) { return Array.from(s); }
  function jsChunk(text, size, overlap) {
    size = size || P.CHUNK_SIZE;
    overlap = overlap || P.CHUNK_OVERLAP;
    if (!text) return [];
    var lines = text.split('\n').map(function (l) { return l.trim(); }).filter(Boolean);
    if (!lines.length) return [];
    var blocks = [], buf = '';
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      if (ln.charAt(0) === '#' && buf) { blocks.push(buf); buf = ''; }
      if (cps(buf).length + cps(ln).length + 1 > size && buf) { blocks.push(buf); buf = ln; }
      else { buf = (buf + ' ' + ln).trim(); }
    }
    if (buf) blocks.push(buf);
    var win = [];
    blocks.forEach(function (b) {
      var cs = cps(b);
      if (cs.length <= size) { win.push(b); return; }
      for (var s = 0; s < cs.length; s += size - overlap) win.push(cs.slice(s, s + size).join(''));
    });
    return win.map(function (b) { return b.replace(/^#+\s*/, '').trim(); }).filter(Boolean);
  }

  /* ---------- 攻防规则（模式串与打包侧同源，见 tools/attack_samples.py） ---------- */
  var R1RE = A ? A.patterns.map(function (p) { return new RegExp(p, 'i'); }) : [];
  function r1Hit(t) {
    for (var i = 0; i < R1RE.length; i++) if (R1RE[i].test(t)) return i;
    return null;
  }
  var NORMRE = A ? new RegExp(A.normClass, 'g') : null;
  function norm(s) { return s.toLowerCase().replace(NORMRE, ''); }
  function r5Where(t, qn, L) {
    var tn = norm(t);
    for (var s = 0; s + L <= qn.length; s++) if (tn.indexOf(qn.slice(s, s + L)) >= 0) return s;
    return -1;
  }
  var PTXTS = A ? A.poison.map(function (p) { return p.text; }) : [];
  var PSRCS = A ? A.poison.map(function (p) { return p.source; }) : [];
  var PDOCS = A ? A.poison.map(function (p) { return p.doc; }) : [];
  var FULL_T = TEXTS.concat(PTXTS);
  var FULL_S = SRC.concat(PSRCS);
  var FULL_D = new Array(N).fill(null).concat(PDOCS);
  var COMBO = A ? A.combo : { r1: false, r2: false, L: 0 };

  /* ---------- 视图：一次检索真正看到的候选集 ----------
     mode = off 干净语料 ／ atk 语料+投毒块 ／ def 投毒块+防御规则过滤
     r2only 只为扫描表「仅堆砌降权」那一行服务 */
  function decorate(o) {
    o.lower = o.texts.map(function (t) { return t.toLowerCase(); });
    o.norms = o.needNorm ? o.texts.map(norm) : null;
    o.n = o.texts.length;
    o.count = {};
    for (var i = 0; i < o.srcs.length; i++) o.count[o.srcs[i]] = (o.count[o.srcs[i]] || 0) + 1;
    o.pen = null;
    if (P.SRC_PENALTY > 0) {
      var vals = Object.keys(o.count).map(function (k) { return o.count[k]; })
        .sort(function (a, b) { return a - b; });
      var med = vals[Math.floor(vals.length / 2)];        // 对齐 sorted(...)[len//2]
      o.pen = {};
      Object.keys(o.count).forEach(function (s) {
        o.pen[s] = Math.min(1, Math.pow(med / o.count[s], P.SRC_PENALTY));
      });
    }
    o.blocked = o.blocked || [];
    return o;
  }
  function makeView(mode, q, pi) {
    if (mode === 'off') {
      return decorate({ mode: 'off', texts: TEXTS, srcs: SRC, docs: new Array(N).fill(null),
        sem: pi < 0 ? null : PRESETS[pi].scores.slice() });
    }
    var texts = FULL_T, srcs = FULL_S, docs = FULL_D;
    var sem = pi < 0 ? null : PRESETS[pi].scores.concat(PRESETS[pi].atk || []);
    var r2 = mode === 'r2only' || (mode === 'def' && COMBO.r2);
    var useR1 = mode === 'def' && COMBO.r1;
    var useR5 = mode === 'def' && COMBO.L > 0 && !!q;
    var blocked = [];
    if (useR1 || useR5) {
      var qn = useR5 ? norm(q) : '';
      var kT = [], kS = [], kD = [], kM = [];
      for (var i = 0; i < texts.length; i++) {
        if (useR1) {
          var h = r1Hit(texts[i]);
          if (h !== null) {
            blocked.push({ t: texts[i], s: srcs[i], d: docs[i], rule: 'R1',
              why: '命中载荷结构特征第 ' + (h + 1) + ' 条 /' + A.patterns[h] + '/' });
            continue;
          }
        }
        if (useR5 && qn.length >= COMBO.L) {
          var at = r5Where(texts[i], qn, COMBO.L);
          if (at >= 0) {
            blocked.push({ t: texts[i], s: srcs[i], d: docs[i], rule: 'R5',
              why: '块内出现提问原句连续 ' + COMBO.L + ' 字（"' + qn.slice(at, at + COMBO.L) + '"）' });
            continue;
          }
        }
        kT.push(texts[i]); kS.push(srcs[i]); kD.push(docs[i]);
        if (sem) kM.push(sem[i]);
      }
      texts = kT; srcs = kS; docs = kD; sem = sem ? kM : null;
    }
    return decorate({ mode: mode, texts: texts, srcs: srcs, docs: docs, sem: sem,
      r2: r2, blocked: blocked, needNorm: useR5 });
  }
  // 自定义语料：只有字面通道。静态页跑不了 embedding，绝不假装能算语义分
  function makeCustomView(texts) {
    return decorate({ mode: 'custom', texts: texts, needNorm: false,
      srcs: texts.map(function (_, i) { return '我的语料 · 第 ' + (i + 1) + ' 块'; }),
      docs: null, sem: null });
  }
  /* 候选集由（模式, 提问）唯一决定，内嵌语料又是只读的 → 算一次就缓存 */
  var VCACHE = {};
  function getView(mode, q, pi) {
    var key = mode + '|' + (pi < 0 ? 'q' + q : 'p' + pi);
    if (!VCACHE[key]) VCACHE[key] = makeView(mode, pi < 0 ? q : PRESETS[pi].query, pi);
    return VCACHE[key];
  }

  /* ---------- 字面腿：与参考实现同公式，作用于任意视图 ---------- */
  function countOcc(hay, needle) {
    var n = 0, i = 0;
    while ((i = hay.indexOf(needle, i)) >= 0) { n++; i += needle.length; }
    return n;
  }
  function median(arr) {                       // 对齐 statistics.median（偶数取均值）
    var v = arr.slice().sort(function (a, b) { return a - b; });
    if (!v.length) return 0;
    var m = Math.floor(v.length / 2);
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
  }
  function keywordLeg(V, tokens) {
    var n = V.n;
    var out = { tokens: tokens, df: {}, idf: {}, raw: new Array(n).fill(0),
      scores: new Array(n).fill(0), per: [], max: 0 };
    if (!tokens.length) return out;
    var i, t, c;
    for (i = 0; i < tokens.length; i++) {
      t = tokens[i];
      var df = 0;
      for (c = 0; c < n; c++) if (V.lower[c].indexOf(t) >= 0) df++;
      out.df[t] = df;
      out.idf[t] = Math.log(1 + (n - df + 0.5) / (df + 0.5));
    }
    for (c = 0; c < n; c++) {
      var s = 0, contrib = {};
      for (i = 0; i < tokens.length; i++) {
        t = tokens[i];
        var cnt = countOcc(V.lower[c], t);
        if (cnt) { var w = out.idf[t] * (1 + Math.log(cnt)); s += w; contrib[t] = { cnt: cnt, w: w }; }
      }
      out.raw[c] = s;
      out.per[c] = contrib;
    }
    var mx = Math.max.apply(null, out.raw);
    for (c = 0; c < n; c++) {
      out.scores[c] = mx > 0 ? out.raw[c] / mx : 0;
      if (V.pen) out.scores[c] *= V.pen[V.srcs[c]];   // 惩罚必须乘在归一化之后
    }
    out.max = mx;
    return out;
  }

  /* ---------- 排名：语义（真值）/字面/降级三态 ---------- */
  function rankBy(V, final) {
    var order = [];
    for (var i = 0; i < V.n; i++) order.push(i);
    order.sort(function (a, b) { return final[b] - final[a] || a - b; });
    var picked = [], dropped = [], quota = {};
    for (var k = 0; k < order.length && picked.length < TOPK; k++) {
      var idx = order[k], s = V.srcs[idx];
      if ((quota[s] || 0) >= P.MAX_PER_SOURCE) {
        dropped.push({ i: idx, final: final[idx] });
        continue;
      }
      quota[s] = (quota[s] || 0) + 1;
      picked.push({ i: idx, final: final[idx] });
    }
    return { picked: picked, dropped: dropped };
  }

  // 一个视图 + 一组关键词 → 字面分、最终分、堆砌降权归因、排名
  function scoreView(V, tokens) {
    var kw = keywordLeg(V, tokens);
    var n = V.n, final = new Array(n), ruled = new Array(n).fill(null);
    for (var i = 0; i < n; i++) {
      final[i] = V.sem ? ((1 - P.KW_WEIGHT) * V.sem[i] + P.KW_WEIGHT * kw.scores[i]) : kw.scores[i];
    }
    if (V.r2 && V.sem) {                       // R2 堆砌降权：MAD 稳健 z
      var gaps = new Array(n);
      for (i = 0; i < n; i++) gaps[i] = kw.scores[i] - V.sem[i];
      var med = median(gaps);
      var mad = median(gaps.map(function (g) { return Math.abs(g - med); }));
      for (i = 0; i < n; i++) {
        var z = 0.6745 * (gaps[i] - med) / (mad || 1e-9);
        if (z > P.DROP_Z) {
          final[i] *= (A ? A.dropFactor : 0.5);
          ruled[i] = { z: z };
        }
      }
    }
    return { kw: kw, final: final, ruled: ruled, rank: rankBy(V, final) };
  }


  /* ---------- 分词 ---------- */
  function freeTokens(q) {
    var set = new Set();
    (q.match(/[A-Za-z_][A-Za-z0-9_]+/g) || []).forEach(function (w) { set.add(w.toLowerCase()); });
    (q.match(/[一-龥]+/g) || []).forEach(function (seg) {
      if (seg.length < 2) return;
      for (var i = 0; i + 2 <= seg.length; i++) {
        var g = seg.slice(i, i + 2);
        if (!STOP.has(g)) set.add(g);
      }
    });
    return Array.from(set);
  }

  /* ---------- 高亮（先转义再包标记，语料里可能有不可信文本） ---------- */
  function esc(s) {
    return s.replace(/[&<>"']/g, function (m) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m];
    });
  }
  function highlight(text, tokens) {
    var real = tokens.filter(function (t) { return t && text.toLowerCase().indexOf(t) >= 0; })
      .sort(function (a, b) { return b.length - a.length; });
    if (!real.length) return esc(text);
    var re = new RegExp('(' + real.map(function (t) {
      return t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }).join('|') + ')', 'gi');
    var buf = '', last = 0, m;
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      if (m.index === re.lastIndex) { re.lastIndex++; continue; }
      buf += esc(text.slice(last, m.index)) + '<mark>' + esc(m[0]) + '</mark>';
      last = m.index + m[0].length;
    }
    return buf + esc(text.slice(last));
  }

  /* ---------- 状态 ----------
     atk: off | atk | def      注入攻击演示开关
     corpus: builtin | mine    换我的语料开关（自定义语料只在内存里，不落盘） */
  var state = {
    mode: 'preset', presetIdx: 0, query: '', set: 'all', ranEval: false,
    atk: 'off', corpus: 'builtin', customQ: ''
  };
  var CUSTOM = { raw: '', texts: [], n0: 0, applied: false };

  function activeQuery() {
    return state.mode === 'preset' ? PRESETS[state.presetIdx].query : state.query;
  }
  function activeTokens() {
    return state.mode === 'preset' ? PRESETS[state.presetIdx].tokens : freeTokens(state.query);
  }
  function setPresets(set) {
    var ids = [];
    PRESETS.forEach(function (p, i) { if (set === 'all' || p.group === set) ids.push(i); });
    return ids;
  }
  // 自定义语料模式下，攻击演示与评测集都无意义（投毒样本与 targets 对应内嵌语料）
  function attackOn() { return !!A && state.corpus === 'builtin' && state.atk !== 'off'; }

  /* ---------- 评测（用真值语义分，与参考实现同口径） ----------
     语料与参数都是打包好的只读数据，同一评测集 + 同一模式的结果恒定，算一次就缓存。 */
  var EV_CACHE = {};
  function evalSet(set, mode) {
    mode = mode || 'off';
    var ck = set + '|' + mode;
    if (EV_CACHE[ck]) return EV_CACHE[ck];
    var ids = setPresets(set), rows = [], hit = 0, hit1 = 0, mrr = 0, dist = 0, dup = 0;
    var poison = 0, blockedN = 0, colN = 0;
    ids.forEach(function (i) {
      var p = PRESETS[i];
      var V = getView(mode, p.query, i);
      var s = scoreView(V, p.tokens);
      var srcs = s.rank.picked.map(function (x) { return V.srcs[x.i]; });
      var docs = s.rank.picked.map(function (x) { return V.docs ? V.docs[x.i] : null; });
      var ruled = s.rank.picked.map(function (x) { return s.ruled[x.i]; });
      var rank = 0;
      for (var k = 0; k < srcs.length; k++) if (p.targets.indexOf(srcs[k]) >= 0) { rank = k + 1; break; }
      if (rank) { hit++; if (rank === 1) hit1++; mrr += 1 / rank; }
      var nd = new Set(srcs).size; dist += nd; if (nd < srcs.length) dup++;
      var pz = 0;
      srcs.forEach(function (x) { if (x.indexOf('99-投毒样本') === 0) pz++; });
      poison += pz; blockedN += V.blocked.length;
      colN += V.blocked.filter(function (b) { return !b.d; }).length;
      rows.push({
        i: i, query: p.query, group: p.group, targets: p.targets, srcs: srcs,
        docs: docs, ruled: ruled, rank: rank, distinct: nd, poison: pz,
        blocked: V.blocked, pool: V.n
      });
    });
    var n = rows.length || 1;
    EV_CACHE[ck] = {
      rows: rows, n: rows.length, mode: mode,
      hit3: hit, hit1: hit1, mrr: mrr / n, distinct: dist / n, dup: dup,
      poison: poison, blocked: blockedN, collateral: colN / n,
      pct: Math.round((hit / n) * 100),
    };
    return EV_CACHE[ck];
  }
  /* ---------- 全池原始位次（只为「逐题位次差值」那张表读一次） ----------
     与 hit@3 用的位次区分开：evalSet 的 rows[].rank 是 top-3 返回列表里的名次（出榜＝0），
     这里要的是「同一份 final 数组、同一个比较器，排完之后正确答案落在第几位」——
     不截断、不套同源配额，纯读取。它不改任何分数、不改排名结果、不写进产物数据，
     算一次就缓存在内存里，攻防面板关掉也不会留痕迹。 */
  var RAW_RANK = {};
  function rawRank(mode, i) {
    var key = mode + '|' + i;
    if (RAW_RANK[key] != null) return RAW_RANK[key];
    var p = PRESETS[i];
    var V = getView(mode, p.query, i);
    var s = scoreView(V, p.tokens);
    var order = [];
    for (var k = 0; k < V.n; k++) order.push(k);
    order.sort(function (a, b) { return s.final[b] - s.final[a] || a - b; });
    var at = 0;
    for (k = 0; k < order.length; k++) {
      if (p.targets.indexOf(V.srcs[order[k]]) >= 0) { at = k + 1; break; }
    }
    RAW_RANK[key] = at;
    return at;
  }

  /* ---------- 与参考实现的一致性自检 ---------- */
  function consistency(mode) {
    mode = mode || 'off';
    if (mode === 'off') {
      var ref = (DATA.baseline && DATA.baseline.main && DATA.baseline.main.rows) || [];
      if (!ref.length) return { ok: null };
      var mine = evalSet('all', 'off').rows;
      var bad = [];
      ref.forEach(function (r, k) {
        var page = mine[k];
        if (!page) return;
        if (page.srcs.join('|') !== r.srcs.join('|')) bad.push({ q: r.query, ref: r.srcs, page: page.srcs });
      });
      return { ok: bad.length === 0, total: ref.length, bad: bad };
    }
    // 攻防态：拿打包时由参考实现算好的期望值逐题比对
    var want = A && A.expect && A.expect[mode === 'def' ? 'def' : mode];
    if (!want) return { ok: null };
    var rows = evalSet('all', mode).rows, bad2 = [];
    rows.forEach(function (r, k) {
      if (r.srcs.join('|') !== want[k].join('|')) {
        bad2.push({ q: r.query, ref: want[k], page: r.srcs });
      }
    });
    return { ok: bad2.length === 0, total: rows.length, bad: bad2 };
  }


  /* ---------- 渲染：预置题列表 ---------- */
  function renderPresets() {
    var ev = evalSet('all', 'off');
    var byIdx = {};
    ev.rows.forEach(function (r) { byIdx[r.i] = r; });
    var evA = attackOn() ? evalSet('all', 'atk') : null;
    var evD = state.atk === 'def' ? evalSet('all', 'def') : null;
    var ids = setPresets(state.set);
    var html = '';
    ids.forEach(function (i) {
      var p = PRESETS[i], r = byIdx[i];
      var on = state.mode === 'preset' && state.presetIdx === i;
      var cls = !r.rank ? 'no' : (r.rank === 1 ? 'ok' : 'mid');
      var label = r.rank ? ('#' + r.rank) : ('未进前' + TOPK);
      if (state.corpus === 'mine') {
        html += '<li><button type="button" class="pbtn" disabled>' +
          '<span class="grp g' + p.group + '">' + p.group + '</span><span class="qt">' + esc(p.query) + '</span>' +
          '<span class="tgt">自定义语料下不可用</span><span class="rankdot no">—</span></button></li>';
        return;
      }
      var extra = '';
      if (evA) {
        var ra = evA.rows[i], rd = evD ? evD.rows[i] : null;
        extra = '<span class="tgt">攻防：投毒进榜 ' + ra.poison + ' 块' +
          (rd ? (' → 防御剔掉 ' + rd.blocked.length + ' 块') : '') + '</span>';
      }
      html += '<li><button type="button" class="pbtn' + (on ? ' on' : '') + '" data-pi="' + i + '">' +
        '<span class="grp g' + p.group + '">' + p.group + '</span>' +
        '<span class="qt">' + esc(p.query) + '</span>' +
        '<span class="tgt">' + esc(p.targets[0].replace(/\.md$/, '')) + '</span>' + extra +
        '<span class="rankdot ' + cls + '">' + label + '</span>' +
        '</button></li>';
    });
    $('presetList').innerHTML = html;
    Array.prototype.forEach.call($('presetList').querySelectorAll('.pbtn:not([disabled])'), function (b) {
      b.addEventListener('click', function () {
        state.mode = 'preset';
        state.presetIdx = +b.getAttribute('data-pi');
        $('q').value = '';
        log('切换预置题', PRESETS[state.presetIdx].query);
        renderAll();
      });
    });
  }

  /* ---------- 渲染：一条命中 ---------- */
  function hitHtml(V, s, toks, targets, p, k, note) {
    var src = V.srcs[p.i], text = V.texts[p.i];
    var isT = targets.indexOf(src) >= 0;
    var doc = V.docs ? V.docs[p.i] : null;
    var poison = !!doc;
    var kw = s.kw, contrib = kw.per[p.i] || {}, semV = V.sem ? V.sem[p.i] : null;
    var rowsTok = '';
    toks.forEach(function (t) {
      var d = contrib[t];
      var zero = !d;
      rowsTok += '<tr class="' + (zero ? 'zero' : '') + '"><td class="tok">' + esc(t) + '</td>' +
        '<td>' + (kw.df[t] != null ? kw.df[t] : '—') + '</td>' +
        '<td>' + (kw.df[t] != null ? kw.idf[t].toFixed(3) : '—') + '</td>' +
        '<td>' + (d ? d.cnt : 0) + '</td>' +
        '<td>' + (d ? d.w.toFixed(3) : '0.000') + '</td></tr>';
    });
    if (!rowsTok) rowsTok = '<tr class="zero"><td class="tok">（无有效关键词）</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>';

    var penV = V.pen ? V.pen[src] : 1;
    var badges = '';
    if (poison) {
      badges += '<span class="badge bad">投毒样本 ' + esc(doc) + '</span>';
      if (V.mode === 'def') badges += '<span class="badge ok">防御后仍在榜：R1/R5 认不出它</span>';
    }
    if (s.ruled[p.i]) {
      badges += '<span class="badge warn">R2 堆砌降权 ×' + (A ? A.dropFactor : 0.5) +
        '（稳健 z=' + s.ruled[p.i].z.toFixed(2) + '）</span>';
    }
    if (note) badges += note;
    return '<article class="hit' + (isT ? ' target' : '') + (poison ? ' poison' : '') + '">' +
      '<div class="hit-top">' +
      '<span class="hit-ord">#' + (k + 1) + (isT ? ' <i class="hitmark">✓ 命中</i>' : '') + '</span>' +
      '<span class="hit-src' + (poison ? ' dead' : '') + '" title="' + esc(src) + '">' + esc(src) + '</span>' +
      '<span class="hit-score">最终 <b>' + p.final.toFixed(4) + '</b></span>' +
      '</div>' +
      '<div class="hit-text"><div class="snip">' + highlight(text, toks) + '</div></div>' +
      (badges ? '<div class="badges">' + badges + '</div>' : '') +
      '<div class="why">' +
      '<div class="why-grid">' +
      cell('语义分（离线实算）', semV == null ? '不可用' : semV.toFixed(4)) +
      cell('字面分（本页计算）', kw.scores[p.i].toFixed(4)) +
      cell('权重合成', semV == null
        ? '降级：0×语义 + 1×字面'
        : '(1−' + P.KW_WEIGHT + ')×语义 + ' + P.KW_WEIGHT + '×字面') +
      cell('本篇块数 / 宽度惩罚', V.count[src] + ' 块 · ×' + penV.toFixed(3)) +
      '</div>' +
      '<div class="bars">' +
      bar('sem', '语义', semV == null ? 0 : Math.max(0, Math.min(1, semV)), semV == null ? '—' : semV.toFixed(3)) +
      bar('kw', '字面', Math.max(0, Math.min(1, kw.scores[p.i])), kw.scores[p.i].toFixed(3)) +
      '</div>' +
      '<div class="toks"><div class="toks-h">' + (V.mode === 'custom'
        ? '清洗后的关键词 → 在你这段文本里的命中明细（df＝出现在几个块里）'
        : '清洗后的关键词 → 在本块的命中明细（df＝全库出现该词的块数，IDF 越罕见越高）') + '</div>' +
      '<table class="toktable"><thead><tr><th>关键词</th><th>df</th><th>IDF</th><th>本块次数</th><th>贡献分</th></tr></thead>' +
      '<tbody>' + rowsTok + '</tbody></table></div>' +
      '<p class="dedupnote">同源去重：本篇配额 ' + P.MAX_PER_SOURCE + ' 块，' +
      '<b>本条已入选</b>；' + (s.rank.dropped.length
        ? '因本篇名额已满被让出的更高分块 <b class="drop">' + s.rank.dropped.filter(function (d) { return V.srcs[d.i] === src; }).length + '</b> 个（见下方展开）'
        : '无块因该规则被过滤') + '。</p>' +
      '</div></article>';
  }

  function inlineDetails(V, s) {
    var out = '';
    if (s.rank.dropped.length) {
      out += '<details class="dropped"><summary>被同源去重过滤掉的更高分块 <span class="cnt">' +
        s.rank.dropped.length + '</span></summary><div class="dropped-body">' +
        s.rank.dropped.slice(0, 8).map(function (d) {
          return '<div class="dline"><span class="s">' + d.final.toFixed(4) + '</span>' +
            '<span class="n">' + esc(V.srcs[d.i]) + ' · ' + esc(V.texts[d.i].slice(0, 60)) + '</span>' +
            '<span class="x">同篇已有更高分块入选</span></div>';
        }).join('') + '</div></details>';
    }
    if (V.blocked.length) {
      out += '<details class="dropped"><summary>被防御规则拦在候选集外 <span class="cnt">' +
        V.blocked.length + '</span></summary><div class="dropped-body">' +
        V.blocked.slice(0, 12).map(function (b) {
          return '<div class="dline"><span class="s">' + b.rule + '</span>' +
            '<span class="n">' + esc(b.s) + ' · ' + esc(b.t.slice(0, 52)) + '</span>' +
            '<span class="x">' + esc(b.why) + '</span></div>';
        }).join('') + '</div></details>';
    }
    return out;
  }

  function emptyHtml(toks, kw) {
    return '<div class="empty">' + (toks.length
      ? '这串问题在全库没有任何字面命中（所有块字面分为 0），排出来也只是随机顺序，所以不展示。' +
        '试试带英文标识符（zip / rglob / islower）或笔记里出现过的中文词，或直接点上方预置题看真实语义检索。'
      : '分词后没有留下可匹配的关键词（全是虚词或标点）。' +
        '试试带英文标识符（zip / rglob / islower）或笔记里出现过的中文词。') + '</div>';
  }

  /* 自由输入（降级通道）的完整口径说明：renderResults 与 renderAttackResults 两处共用，
     只追加文案，不参与任何打分与排名。 */
  var FREE_INPUT_WARN = '这条结果不是语义检索：纯静态页跑不动 embedding 模型，你现打的这句话没有向量可算。' +
    '预置题的语义余弦是离线实算后固化进产物的，现场无法为新问题补算；' +
    '而线上取值 KW_WEIGHT=' + P.KW_WEIGHT + '，字面腿在评测口径里权重为 0——' +
    '所以下面这个排名来自一条「线上不参与排序」的通道，它只用来看词怎么命中，不能与上方预置题的结果同表比较。';

  /* ---------- 渲染：结果 ---------- */
  function renderResults() {
    if (state.corpus === 'mine') { renderCustomResults(); return; }
    var q = activeQuery();
    var toks = activeTokens();
    var pi = state.mode === 'preset' ? state.presetIdx : -1;
    var targets = state.mode === 'preset' ? PRESETS[pi].targets : [];
    $('atkPanel').hidden = !attackOn();

    if (attackOn()) { renderAttackResults(q, toks, pi, targets); return; }

    $('trioBox').hidden = true;
    $('results').className = 'results';
    var V = getView('off', q, pi);
    var s = scoreView(V, toks);
    var sem = V.sem;

    $('channelTag').className = 'tag' + (sem ? '' : ' warn');
    $('channelTag').textContent = sem ? '语义通道 · 真值' : '降级 · 仅字面';
    $('channelTxt').textContent = sem
      ? '本题语义分为离线用 ' + DATA.provenance.model + ' 实算的余弦值；KW_WEIGHT=' + P.KW_WEIGHT +
        '，故排名只看语义，字面分仅作参考值一并显示。'
      : '纯静态页无法现场跑 embedding 模型，自由输入不产生语义分，以下结果只由字面通道（IDF+词频饱和+宽度惩罚）排出，' +
        '且中文按 2-gram 近似切分。要看真实语义检索，请点上方预置题。' + FREE_INPUT_WARN;

    $('resSub').textContent = '提问：' + q + ' ｜ 候选 ' + V.n + ' 块 / ' + DATA.meta.sources + ' 篇';

    // 空信号：没有可用关键词，或所有块的字面分都是 0 —— 这时排名是任意的，不能装成结果
    var noSignal = !toks.length || (sem ? false : !(s.kw.max > 0));
    if (!s.rank.picked.length || noSignal) {
      $('results').innerHTML = emptyHtml(toks, s.kw);
      $('droppedBox').style.display = 'none';
      $('droppedCnt').textContent = '0';
      $('blockedBox').style.display = 'none';
      return;
    }
    $('results').innerHTML = s.rank.picked.map(function (p, k) {
      return hitHtml(V, s, toks, targets, p, k);
    }).join('');

    /* 被过滤掉的更高分块 */
    $('droppedBox').style.display = s.rank.dropped.length ? '' : 'none';
    $('droppedCnt').textContent = s.rank.dropped.length;
    $('droppedBody').innerHTML = s.rank.dropped.slice(0, 8).map(function (d) {
      return '<div class="dline"><span class="s">' + d.final.toFixed(4) + '</span>' +
        '<span class="n">' + esc(SRC[d.i]) + ' · ' + esc(TEXTS[d.i].slice(0, 60)) + '</span>' +
        '<span class="x">同篇已有更高分块入选</span></div>';
    }).join('');
    $('blockedBox').style.display = 'none';
  }

  /* 攻击态：并排展示 攻击前 / 攻击后 / 防御后 三个视图（防御档才有三格） */
  function renderAttackResults(q, toks, pi, targets) {
    var modes = state.atk === 'def' ? ['off', 'atk', 'def'] : ['atk'];
    var labels = {
      off: ['攻击前（干净语料）', '不含任何投毒样本 —— 与开关关掉时看到的同一份结果'],
      atk: ['攻击后（语料里混入投毒块）', '红框即投毒样本，看它把哪条真答案挤了出去'],
      def: ['防御后（' + A.defCombo + ' 过滤后再排）', '绿标说明这一条是投毒块被拦下后回补上来的']
    };
    var html = '';
    var last = null;
    var atkPicked = null;
    if (pi < 0) {
      html += '<div class="warnline">这串问题是自由输入：静态页跑不了 embedding，没有语义真值，' +
        '下面三栏只能按字面排，<b>不代表真实攻防效果</b>。' +
        '点上方任意一道预置题（它们的语义分是离线用 ' + DATA.provenance.model + ' 实算的）就能看到真实对比。</div>';
    }
    modes.forEach(function (m) {
      var V = getView(m, q, pi);
      var s = scoreView(V, toks);
      last = { V: V, s: s };
      if (m === 'atk') atkPicked = s.rank.picked.map(function (p) { return p.final; });
      var semless = !V.sem;
      var noSignal = !toks.length || (semless ? !(s.kw.max > 0) : false);
      html += '<div class="blk"><h3 class="blk-h">' + labels[m][0] +
        '<span class="muted small"> ｜ 候选 ' + V.n + ' 块 · ' + labels[m][1] + '</span></h3>' +
        (noSignal || !s.rank.picked.length
          ? '<div class="empty">' + (pi < 0
            ? '自由输入没有语义分，攻击演示请用上方预置题（离线真值语义分才成立）。'
            : emptyHtml(toks, s.kw)) + '</div>'
          : s.rank.picked.map(function (p, k) {
            var nb = '';
            if (m === 'off') nb = '<span class="badge">与关开关时同一份结果</span>';
            else if (m === 'atk' && !V.docs[p.i]) nb = '<span class="badge bad">攻击态第 ' + (k + 1) + ' 名</span>';
            else if (m === 'def') {
              nb = '<span class="badge">' + A.defCombo + ' 剔 ' + V.blocked.length + ' 块后再排</span>';
              if (!V.docs[p.i] && (!atkPicked || atkPicked.indexOf(p.final) < 0)) {
                nb += '<span class="badge ok">防御回补：投毒被剔后补进前 ' + TOPK + '</span>';
              }
            }
            return hitHtml(V, s, toks, targets, p, k, nb);
          }).join('') + inlineDetails(V, s)) +
        '</div>';
    });
    $('results').innerHTML = html;
    $('results').className = 'results' + (modes.length > 1 ? ' has-blk' : '');

    // 通道说明：攻击演示只在预置题（有语义真值）下有意义
    $('channelTag').className = 'tag' + (pi < 0 ? ' warn' : '');
    $('channelTag').textContent = pi < 0 ? '降级 · 仅字面' : '语义通道 · 真值';
    $('channelTxt').textContent = pi < 0
      ? '当前是自由输入：没有语义分，投毒块只能按字面参与排名，下面的对比不代表真实攻击效果。请点上方预置题看攻防演示。' + FREE_INPUT_WARN
      : '攻防演示只改「候选集」，不改参数也不改内嵌语料：投毒块作为额外的块参与同一套排名，' +
        '防御档则在排名前按 ' + A.defCombo + ' 把可疑块剔出候选集。三个视图各自独立，关掉开关立刻回到原始结果。';
    $('resSub').textContent = '提问：' + q + ' ｜ 攻击演示：' + A.poison.length + ' 个投毒块' +
      (state.atk === 'def' ? ' + 防御规则 ' + A.defCombo : '');
    $('droppedBox').style.display = 'none';
    $('blockedBox').style.display = 'none';
    renderTrio(pi, last);
  }

  /* 攻击前 / 攻击后 / 防御后 三格 hit@3 */
  function renderTrio() {
    var box = $('trioBox');
    if (!attackOn() || state.atk !== 'def') { box.hidden = true; return; }
    var evB = evalSet(state.set, 'off'), evA = evalSet(state.set, 'atk'), evD = evalSet(state.set, 'def');
    var cards = [
      { t: '攻击前 hit@3', v: evB.hit3 + '/' + evB.n, x: '干净语料，与顶部大分数同一个数' },
      { t: '攻击后 hit@3', v: evA.hit3 + '/' + evA.n, x: '混入 ' + A.poison.length + ' 个投毒块：掉 ' +
        (evB.hit3 - evA.hit3) + ' 题，投毒占走 ' + evA.poison + ' 个名额', dir: evA.hit3 < evB.hit3 ? 'down' : '' },
      { t: '防御后 hit@3', v: evD.hit3 + '/' + evD.n, x: A.defCombo + '：投毒名额剩 ' + evD.poison +
        ' 个，误伤正常笔记 ' + evD.collateral.toFixed(1) + ' 块/题', dir: evD.hit3 > evA.hit3 ? 'up' : '' }
    ];
    box.innerHTML = cards.map(function (c, i) {
      return '<div class="t' + (i === cards.length - 1 ? ' cur' : '') + (c.dir ? ' ' + c.dir : '') + '">' +
        '<div class="tl">' + c.t + '</div><div class="tv num">' + c.v + '</div>' +
        '<div class="tx">' + esc(c.x) + '</div></div>';
    }).join('');
    box.hidden = false;
  }

  /* ---------- 渲染：访客自带语料 ---------- */
  function renderCustomResults() {
    $('trioBox').hidden = true;
    $('results').className = 'results';
    $('atkPanel').hidden = true;
    $('blockedBox').style.display = 'none';
    var q = state.customQ;
    renderCorpusPanel();
    if (!CUSTOM.texts.length) {
      $('channelTag').className = 'tag warn';
      $('channelTag').textContent = '自定义语料';
      $('channelTxt').textContent = '这一步不改变评测集：上面的文本还没切成块。' +
        '先在粘贴框贴一段话，再点「用这段检索」——切好的块会显示在下面。';
      $('resSub').textContent = '语料：等待你的文本（临时，刷新即恢复）';
      $('results').innerHTML = '<div class="empty">还没有可用语料。在上面贴一段文本（几百字即可），点「用这段检索」，' +
        '就能看到这段文本被切成几块、你问的词命中哪一块。</div>';
      $('droppedBox').style.display = 'none';
      return;
    }
    if (!q) {
      $('channelTag').className = 'tag warn';
      $('channelTag').textContent = '自定义语料';
      $('channelTxt').textContent = '上面的文本已切好块，在提问框里问一句就能看到它怎么排。' +
        '这条通道只有字面分：静态页跑不了 embedding 模型，所以这里不显示语义分，也不参与评测集。';
      $('resSub').textContent = '语料：你粘贴的文本 · ' + CUSTOM.texts.length + ' 块（临时，刷新即恢复）';
      $('results').innerHTML = '<div class="empty">还在等你的问题。在提问框里输入关键词再点「检索」——' +
        '建议直接用上面某块里出现过的词。</div>';
      $('droppedBox').style.display = 'none';
      return;
    }
    var toks = freeTokens(q);
    var V = makeCustomView(CUSTOM.texts);
    var s = scoreView(V, toks);
    $('channelTag').className = 'tag warn';
    $('channelTag').textContent = '降级 · 仅字面';
    $('channelTxt').textContent = '自定义语料走的是纯字面通道（IDF + 词频饱和 + 宽度惩罚），' +
      '中文按 2-gram 近似切分；静态页内跑不了 embedding，这里没有语义分，' +
      '所以这份排名不能和内嵌语料那 ' + N + ' 块的分数放在一张表里比 —— 它是用来看「清洗与分块怎么切、词怎么命中」的。' +
      '另外字面分做了 min-max 归一，命中最好的那块一定是 1.0000，那是归一化的规定动作，不是「满分命中」。';
    $('resSub').textContent = '提问：' + q + ' ｜ 候选 ' + V.n + ' 块（你粘贴的文本，仅本次会话）';
    if (!s.rank.picked.length || !toks.length || !(s.kw.max > 0)) {
      $('results').innerHTML = '<div class="empty">你这段文本里没有匹配上这串问题的词（所有块字面分为 0）。' +
        '换个文本里确实出现过的词试试 —— 上面「切块结果」里能直接看到每块原文。</div>';
      $('droppedBox').style.display = 'none';
      return;
    }
    $('results').innerHTML = s.rank.picked.map(function (p, k) {
      return hitHtml(V, s, toks, [], p, k);
    }).join('') + inlineDetails(V, s);
    $('droppedBox').style.display = 'none';
  }

  function chunkSelfTest() {
    if (!CT || !CT.length) return null;
    var ok = 0;
    var bad = [];
    CT.forEach(function (c, i) {
      var mine = jsChunk(jsClean(c.raw));
      var same = mine.length === c.chunks.length &&
        mine.every(function (x, k) { return x === c.chunks[k]; });
      if (same) ok++; else bad.push(i + 1);
    });
    return { ok: ok, total: CT.length, bad: bad };
  }
  var SELF_CHUNK = chunkSelfTest();

  /* R1 在本库误伤的正常块（不含投毒样本）—— 现场按同一份模式串数出来，不靠打包时手写。
     返回 {n, srcs, pats}：n＝去重后的块数（每块只数一次），srcs＝涉及哪几篇，pats＝被第几条模式命中。
     这三样只用来把「误报率的分母与单位」写清楚，不参与任何排名或分数计算。 */
  var R1_COLLATERAL = (function () {
    var res = { n: 0, srcs: [], pats: [] };
    if (!A) return res;
    for (var i = 0; i < FULL_T.length; i++) {
      if (FULL_D[i]) continue;                        // 投毒块本身不算误伤
      var h = r1Hit(FULL_T[i]);
      if (h === null) continue;
      res.n++;
      if (res.srcs.indexOf(FULL_S[i]) < 0) res.srcs.push(FULL_S[i]);
      if (res.pats.indexOf(h) < 0) res.pats.push(h);
    }
    return res;
  })();
  /* R5 另剔到的正常块总数（跨 10 道题累加）。R5 按提问生效，所以它和 R1 那一批不同口径：
     R1 与提问无关，每道题看到的都是同一批块；R5 只在提问够长的那几道题里额外剔块。
     与 makeView 同一顺序：先 R1 命中就跳过，不会被 R1、R5 重复计数。 */
  var R5_COLLATERAL = (function () {
    if (!A || !COMBO.r1 && !COMBO.L) return 0;
    var n = 0;
    for (var k = 0; k < PRESETS.length; k++) {
      var qn = COMBO.L > 0 ? norm(PRESETS[k].query) : '';
      if (qn.length < COMBO.L) continue;
      for (var i = 0; i < FULL_T.length; i++) {
        if (FULL_D[i]) continue;
        if (COMBO.r1 && r1Hit(FULL_T[i]) !== null) continue;
        if (r5Where(FULL_T[i], qn, COMBO.L) >= 0) n++;
      }
    }
    return n;
  })();

  function renderCorpusPanel() {
    var box = $('corpusChunks'), raw = CUSTOM.raw;
    $('corpusCharN').textContent = cps(raw).length + ' / 4000 字';
    var st = [];
    st.push(SELF_CHUNK
      ? '<span class="badge ' + (SELF_CHUNK.ok === SELF_CHUNK.total ? 'ok' : 'bad') + '">' +
        '分块实现自检 ' + SELF_CHUNK.ok + '/' + SELF_CHUNK.total + ' 逐块一致' +
        (SELF_CHUNK.bad.length ? '（不一致样本 ' + SELF_CHUNK.bad.join(',') + '）' : '') + '</span>'
      : '<span class="badge">分块自检样本未打包</span>');
    if (!raw.trim()) {
      $('corpusState').textContent = '未启用';
      CUSTOM.texts = []; CUSTOM.applied = false;
      box.innerHTML = '<p class="muted small">还没贴文本。' + st.join(' ') + '</p>';
      return;
    }
    var cleaned = jsClean(raw), chunks = jsChunk(cleaned);
    CUSTOM.texts = chunks;
    $('corpusState').textContent = '已切成 ' + chunks.length + ' 块' +
      (CUSTOM.applied ? ' · 正在按你的提问检索' : ' · 等待提问');
    box.innerHTML = '<p class="muted small">' + st.join(' ') +
      ' 原文 ' + cps(raw).length + ' 字 → 清洗后 ' + cps(cleaned).length + ' 字 → ' +
      chunks.length + ' 块（chunk_size=' + P.CHUNK_SIZE + '，overlap=' + P.CHUNK_OVERLAP + '）</p>' +
      '<div class="chunklist">' + chunks.map(function (c, i) {
        var r1 = r1Hit(c);
        return '<div class="chunk"><div class="ch"><span>第 ' + (i + 1) + ' 块</span>' +
          '<span>' + cps(c).length + ' 字' + (r1 !== null ? ' · R1 命中结构特征' : '') + '</span></div>' +
          '<div class="cb">' + esc(c) + '</div></div>';
      }).join('') + '</div>';
  }


  function cell(k, v) { return '<div class="why-cell"><span class="k">' + k + '</span><br><span class="v">' + esc(String(v)) + '</span></div>'; }
  function bar(cls, lab, ratio, val) {
    return '<div class="bar ' + cls + '"><span class="lab">' + lab + '</span>' +
      '<span class="track"><span class="fill" style="width:' + (ratio * 100).toFixed(1) + '%"></span></span>' +
      '<span class="val">' + val + '</span></div>';
  }

  /* ---------- 渲染：顶部分数 ---------- */
  function renderScore() {
    var ev = evalSet(state.set, 'off');
    $('hitScore').textContent = ev.hit3 + '/' + ev.n;
    $('hit1Score').textContent = ev.hit1 + '/' + ev.n;
    $('mrrScore').textContent = ev.mrr.toFixed(3);
    $('corpusSize').textContent = state.corpus === 'mine'
      ? '内嵌 ' + N + ' 块（当前用你的 ' + CUSTOM.texts.length + ' 块）'
      : N + ' 块';
    var box = ($('hitScore').closest && $('hitScore').closest('.score')) || $('scorebar');
    box.classList.remove('flash');
    void box.offsetWidth;
    box.classList.add('flash');
    renderTrio();
  }

  /* ---------- 渲染：评测表 + 对照 ---------- */
  function renderEval() {
    var ev = evalSet(state.set, 'off');
    var b = DATA.baseline || {};
    $('evalPanel').hidden = false;
    $('evalMetricNote').textContent = 'hit@' + TOPK + '＝正确笔记进前 ' + TOPK + '；hit@1＝排第一；MRR＝平均倒数排名' +
      (attackOn() ? '（本表恒为干净语料口径，攻防对照见上方演示面板）' : '');
    $('evalBody').innerHTML = ev.rows.map(function (r) {
      var cls = r.rank ? 'hitrow' : 'miss';
      return '<tr class="' + cls + '"><td class="n"><span class="grp g' + r.group + '">' + r.group + '</span></td><td>' + esc(r.query) + '</td>' +
        '<td class="n">' + esc(r.targets[0]) + '</td>' +
        '<td class="n rank">' + (r.rank || '未命中') + '</td>' +
        '<td class="srcs">' + r.srcs.map(esc).join(' · ') + '</td></tr>';
    }).join('');

    var cards = [
      { t: '线上口径：语义 + 同源去重（本页实时算）', v: ev.hit3 + '/' + ev.n + ' · MRR ' + ev.mrr.toFixed(3), x: '不同来源 ' + ev.distinct.toFixed(2) + ' / ' + P.TOP_K + '，重复题 ' + ev.dup + ' 道', cur: true },
      { t: '参考实现基线（离线跑同一份数据）', v: b.main ? b.main.hit3 + ' · MRR ' + b.main.mrr : '—', x: b.main ? '不同来源 ' + b.main.distinct + '，重复题 ' + b.main.dup : '', cur: false },
      { t: '对照：关闭同源去重', v: b.noDedup ? b.noDedup.hit3 + ' · MRR ' + b.noDedup.mrr : '—', x: b.noDedup ? '重复题 ' + b.noDedup.dup + '（同篇相邻块一起挤进 top-k）' : '', cur: false },
      { t: '对照：纯字面腿排名（KW_WEIGHT=1）', v: b.keywordOnly ? b.keywordOnly.hit3 + ' · MRR ' + b.keywordOnly.mrr : '—', x: b.keywordOnly ? 'hit@1 ' + b.keywordOnly.hit1 + '；与线上取值的取舍见下方说明' : '', cur: false },
    ];
    $('compareBox').innerHTML = cards.map(function (c) {
      return '<div class="cmp' + (c.cur ? ' cur' : '') + '"><div class="ct">' + c.t + '</div>' +
        '<div class="cv num">' + c.v + '</div><div class="cx">' + c.x + '</div></div>';
    }).join('');

    var cons = consistency();
    /* 指路条：B 组探针的出榜位次对比在页脚「数据说明」折叠区里，从这里一键展开+滚动直达。
       只做导航，不复制内容——位次对比面板仍只在页脚渲染一份（避免双份口径各说各话）。 */
    (function ensureProbeJump(){
      var bar = document.getElementById('probeJump');
      if (!bar) {
        bar = document.createElement('p');
        bar.id = 'probeJump';
        bar.className = 'probe-jump';
        bar.innerHTML = '<span>B 组两题是故意设计会输的探针（题面不出现目标笔记关键词）。</span>' +
          '<button type="button" class="pc-toggle" id="probeJumpBtn">看它们的出榜位次对比 →</button>';
        $('compareBox').insertAdjacentElement('afterend', bar);
        bar.querySelector('#probeJumpBtn').addEventListener('click', function () {
          var folds = document.querySelectorAll('footer details.folding');
          var fold = folds[folds.length - 1];   /* 最后一个折叠 = 「这个页面是怎么算出来的 / 数据说明」 */
          if (fold) fold.open = true;
          var target = document.querySelector('.probe-compare');
          if (!target) return;
          target.scrollIntoView({ behavior: 'smooth', block: 'center' });
          target.classList.remove('flash');
          void target.offsetWidth;              /* 重置动画 */
          target.classList.add('flash');
        });
      }
    })();
    if (cons.ok === true) {
      $('compareBox').insertAdjacentHTML('beforeend',
        '<div class="cmp"><div class="ct">一致性自检</div><div class="cv num" style="color:var(--ok)">✓ ' +
        cons.total + '/' + cons.total + ' 题</div><div class="cx">浏览器排名与参考实现 search() 逐题 top-3 及顺序完全相同</div></div>');
    } else if (cons.ok === false) {
      $('compareBox').insertAdjacentHTML('beforeend',
        '<div class="cmp" style="border-color:var(--bad)"><div class="ct">一致性自检</div>' +
        '<div class="cv num" style="color:var(--bad)">✗ ' + cons.bad.length + ' 题不一致</div>' +
        '<div class="cx">' + cons.bad.map(function (x) { return esc(x.q); }).join('；') + '</div></div>');
    }
    $('runState').textContent = '已跑完 · ' + ev.n + ' 题 · ' + new Date().toLocaleTimeString('zh-CN', { hour12: false });
  }

  /* ---------- 渲染：攻防演示面板 ---------- */
  function docRuleFlags(id) {
    var textsOf = A.poison.filter(function (p) { return p.doc === id; }).map(function (p) { return p.text; });
    var r1 = A.poison.some(function (p) { return p.doc === id && p.r1 !== null && p.r1 !== undefined; });
    var r5 = false;
    if (COMBO.L) {
      PRESETS.forEach(function (p) {
        var qn = norm(p.query);
        if (qn.length < COMBO.L || r5) return;
        if (textsOf.some(function (t) { return r5Where(t, qn, COMBO.L) >= 0; })) r5 = true;
      });
    }
    return { r1: r1, r5: r5 };
  }

  function renderAttackPanel() {
    if (!A) return;
    var evA = evalSet('all', 'atk'), evD = evalSet('all', 'def'), evB = evalSet('all', 'off');
    $('atkTopK').textContent = TOPK;
    $('atkPoisonN').textContent = A.poison.length;
    var colPat = R1_COLLATERAL.pats.map(function (k) { return '/' + A.patterns[k] + '/'; }).join('、');
    var colAvg = PRESETS.length
      ? (R1_COLLATERAL.n * PRESETS.length + R5_COLLATERAL) / PRESETS.length : 0;
    $('atkNote').textContent = '本页只做检索层攻防：投毒块只影响「进不进候选集、排第几」，' +
      '不涉及生成。所以看到的污染是名额被抢走，而不是答案被改写。' +
      'R1 的误报有两个口径，别混着读。① 块数口径：' + R1_COLLATERAL.n + ' / ' + N + ' 正常块（' +
      (R1_COLLATERAL.n / N * 100).toFixed(1) + '%），分母是内嵌语料的正常块数，按块去重（一块只数一次）。' +
      '这 ' + R1_COLLATERAL.n + ' 块全部来自 ' + (R1_COLLATERAL.srcs.join('、') || '—') +
      '，且都由 ' + colPat + ' 这一条模式命中 —— 不是 13 条模式各误伤一点，而是一条模式在一篇讲注入实验的笔记里反复命中。' +
      '所以这个 ' + (R1_COLLATERAL.n / N * 100).toFixed(1) + '% 只对本库成立：库里没有这类演示文本时同一批模式可能一条都不中；' +
      '反过来只要库里有讨论攻击的笔记，字面结构检测就必然扫到它。' +
      '② 扫描表里「误伤正常块（块/题）」是另一个口径：每题平均剔除块数，分母是 ' + PRESETS.length + ' 道题。' +
      '本页采用组合的 ' + colAvg.toFixed(1) + ' =（R1 的 ' + R1_COLLATERAL.n + ' 块 × ' + PRESETS.length + ' 题 + R5 另外剔到的 ' +
      R5_COLLATERAL + ' 块）÷ ' + PRESETS.length + ' 题。R1 与提问无关，每题剔的都是同一批 ' +
      R1_COLLATERAL.n + ' 块；R5 按提问生效，只在提问长度 ≥ L=' + A.echoLen + ' 字的题目里额外剔块。' +
      '一个是「一共几块」，一个是「每题平均几块」，数值对不上是正常的。';

    // 进榜次数：每个投毒文档在攻击态／防御态各占了几格
    var inAtk = {}, inDef = {};
    evA.rows.forEach(function (r) {
      r.docs.forEach(function (d) { if (d) inAtk[d] = (inAtk[d] || 0) + 1; });
    });
    evD.rows.forEach(function (r) {
      r.docs.forEach(function (d) { if (d) inDef[d] = (inDef[d] || 0) + 1; });
    });
    $('atkDocBody').innerHTML = A.docs.map(function (d) {
      var f = docRuleFlags(d.id);
      return '<tr><td class="n">' + d.id + '</td><td>' + esc(d.kind) + '</td>' +
        '<td class="srcs">' + esc(d.note) + '<br><span class="muted small">' + esc(d.source) + '</span></td>' +
        '<td class="n">' + d.chunks + '</td>' +
        '<td class="n">' + (f.r1 ? '<span class="badge ok">拦得住</span>' : '<span class="badge">看不见</span>') + '</td>' +
        '<td class="n">' + (f.r5 ? '<span class="badge ok">拦得住</span>' : '<span class="badge">看不见</span>') + '</td>' +
        '<td class="n">攻击 ' + (inAtk[d.id] || 0) + ' 格 → 防御 ' + (inDef[d.id] || 0) + ' 格</td></tr>';
    }).join('');

    $('atkRuleBody').innerHTML = A.defenses.map(function (r) {
      return '<tr><td class="n">' + r.id + ' ' + esc(r.name) + '</td><td>' + esc(r.action) + '</td>' +
        '<td class="srcs">' + esc(r.origin) + '</td><td class="srcs">' + esc(r.weak) + '</td></tr>';
    }).join('');

    $('atkScanBody').innerHTML = A.defenseScan.map(function (s) {
      var on = s.name === A.defCombo;
      return '<tr class="' + (on ? 'hitrow' : '') + '"><td>' + esc(s.name) +
        (s.L ? ' <span class="muted small">L=' + s.L + '</span>' : '') + '</td>' +
        '<td class="n rank">' + s.hit3 + '/' + PRESETS.length + '</td>' +
        '<td class="n">' + s.poisonInTop3 + '</td>' +
        '<td class="n">' + (s.blockedDocs.length ? esc(s.blockedDocs.join(' ')) : '—') + '</td>' +
        '<td class="n">' + s.collateral + '</td>' +
        '<td class="n">' + (on ? '★ 本页采用' : '') + '</td></tr>';
    }).join('');

    // 提问侧：载荷照样进 prompt，这里只看它把检索带偏了多少
    $('atkQuery').innerHTML = A.queries.map(function (qa) {
      var clean = qa.clean;
      var rawHtml = esc(qa.query).replace(esc(clean), '<b>' + esc(clean) + '</b>');
      var rows = [
        ['夹带载荷的提问', '<span class="v txt">' + rawHtml + '</span>', true],
        ['剥掉载荷后', '<span class="v txt">' + esc(clean) + '</span>', true],
        ['真实意图所在', '<span class="v txt">' + esc(qa.targets.join(' / ')) + '</span>', true],
        ['带载荷返回', '<span class="v txt">' + qa.srcs.map(function (s, i) {
          return '#' + (i + 1) + ' ' + esc(s) + (qa.poison[i] ? '（投毒 ' + qa.poison[i] + '）' : '');
        }).join(' → ') + '</span>', true],
        ['剥载荷后返回', '<span class="v txt">' + qa.cleanSrcs.map(esc).join(' → ') + '</span>', true]
      ];
      var hitsRaw = qa.srcs.some(function (s) { return qa.targets.indexOf(s) >= 0; });
      var hitsClean = qa.cleanSrcs.some(function (s) { return qa.targets.indexOf(s) >= 0; });
      return '<div class="qa"><div class="row"><span class="k">类别</span><span class="v txt">' +
        esc(qa.kind) + '</span></div>' +
        rows.map(function (r) {
          return '<div class="row"><span class="k">' + r[0] + '</span>' + r[1] + '</div>';
        }).join('') +
        '<div class="row"><span class="k">结论</span><span class="v txt">' +
        (hitsRaw ? '检索仍命中' : '检索已被带偏') + '；剥载荷后' + (hitsClean ? '命中' : '仍未命中') +
        '。' + esc(qa.note) + 'R1 对这条提问' + (qa.r1Raw !== null && qa.r1Raw !== undefined ? '命中结构特征第 ' + (qa.r1Raw + 1) + ' 条' : '无感') +
        '，但提问侧载荷不进索引，检索层的规则动不了它 —— 只有提问清洗或生成侧防护能拦。</span></div></div>';
    }).join('');

    // 自检：三种状态的排名要与打包时参考实现算的期望值逐题一致
    var okB = checkAgainst(evB.rows, A.expect.base);
    var okA = checkAgainst(evA.rows, A.expect.atk);
    var okD = checkAgainst(evD.rows, A.expect.def);
    var cards = [
      { t: '干净语料 hit@3（参考实现同数据）', v: evB.hit3 + '/' + evB.n, x: '离线基线 ' + A.metrics.base + '/' + PRESETS.length },
      { t: '投毒后 hit@3', v: evA.hit3 + '/' + evA.n, x: '投毒占走 ' + evA.poison + ' 个 top-' + TOPK + ' 名额；离线 ' + A.metrics.atk + '/' + PRESETS.length },
      { t: '防御后 hit@3（' + A.defCombo + '）', v: evD.hit3 + '/' + evD.n, x: '剩 ' + evD.poison + ' 格投毒；误伤正常笔记 ' + evD.collateral.toFixed(1) + ' 块/题；离线 ' + A.metrics.def + '/' + PRESETS.length },
      { t: '攻防一致性自检', v: (okB.ok === okB.total && okA.ok === okA.total && okD.ok === okD.total ? '✓ 30/30' : '✗ 有不一致'), x: '比对的是排名引擎：浏览器排出的名次序列与参考实现 search() 在同一候选集上逐题一致；R1/R5 两条规则本身为本产物实现，参考实现里没有。干净 ' +
        okB.ok + '/' + okB.total + '、投毒 ' + okA.ok + '/' + okA.total + '、防御 ' + okD.ok + '/' + okD.total + ' 题一致', cur: true }
    ];
    $('atkStats').innerHTML = cards.slice(0, 3).map(function (c) {
      return '<div class="cmp"><div class="ct">' + c.t + '</div><div class="cv num">' + c.v + '</div>' +
        '<div class="cx">' + esc(c.x) + '</div></div>';
    }).join('');
    $('atkSelf').innerHTML = cards.slice(3).map(function (c) {
      var good = okB.ok === okB.total && okA.ok === okA.total && okD.ok === okD.total;
      return '<div class="cmp' + (good ? ' cur' : '" style="border-color:var(--bad)') +
        '"><div class="ct">' + c.t + '</div><div class="cv num" style="color:var(--' + (good ? 'ok' : 'bad') + ')">' +
        c.v + '</div><div class="cx">' + esc(c.x) + '</div></div>';
    }).join('');
    renderAtkDelta(evB, evA, evD);
    renderScanNote();
  }

  /* ---------- 逐题位次差值：哪一题掉出前 3、哪一题被补回 ----------
     读的都是页面已有结果：rows[].rank（top-3 返回列表里的名次）、rows[].poison（投毒占格），
     外加 rawRank() 对同一份 final 数组的一次全池读数。
     不改分数、不改排名逻辑、不新增打包数据；面板每次展开只插一次。 */
  function renderAtkDelta(evB, evA, evD) {
    if (document.getElementById('atkDeltaWrap')) return;
    var body = evB.rows.map(function (rb, k) {
      var ra = evA.rows[k], rd = evD.rows[k];
      var rawB = rawRank('off', rb.i), rawA = rawRank('atk', rb.i), rawD = rawRank('def', rb.i);
      var lost = rb.rank > 0 && ra.rank === 0;
      var back = ra.rank === 0 && rd.rank > 0;
      var verdict;
      if (lost && back) {
        verdict = '攻击后掉出前 3（返回位次 ' + rb.rank + ' → 出榜）；防御后补回第 ' + rd.rank + '（投毒占格 ' +
          ra.poison + ' → ' + rd.poison + '）';
      } else if (lost) {
        verdict = '攻击后掉出前 3，防御后仍未回到榜内';
      } else if (!rb.rank && !ra.rank && !rd.rank) {
        verdict = '三档都不在前 3，不参与这组差值';
      } else if (rb.rank === ra.rank && ra.rank === rd.rank && rawB === rawA && rawA === rawD) {
        verdict = '三档完全不变';
      } else if (rb.rank === ra.rank && ra.rank === rd.rank) {
        verdict = '返回位次不变（仍是第 ' + rb.rank + '），只有全池原始位次被挤动，不影响 hit@3';
      } else {
        verdict = '始终在前 3，名次在三档间移动（返回 ' + rb.rank + ' → ' + ra.rank + ' → ' + rd.rank + '）';
      }
      var cls = lost || back ? 'lost' : (!rb.rank && !ra.rank && !rd.rank ? 'miss' : '');
      return '<tr class="' + cls + '"><td class="n">Q' + (k + 1) + '</td><td class="n">' + rb.group + '</td>' +
        '<td>' + esc(rb.query) + '</td><td class="srcs">' + esc(rb.targets.join(' / ')) + '</td>' +
        '<td class="n pos">' + rawB + ' → ' + rawA + ' → ' + rawD + '</td>' +
        '<td class="n pos">' + (rb.rank || '出榜') + ' → ' + (ra.rank || '出榜') + ' → ' + (rd.rank || '出榜') + '</td>' +
        '<td class="n">' + ra.poison + ' → ' + rd.poison + '</td>' +
        '<td class="srcs">' + verdict + '</td></tr>';
    }).join('');
    $('atkStats').insertAdjacentHTML('afterend',
      '<div class="tablewrap" id="atkDeltaWrap">' +
      '<h3 class="sh atk-sub">逐题位次差值（关闭 → 攻击 → 防御）</h3>' +
      '<table class="tbl"><caption class="visually-hidden">每道题的正确答案位次在三档之间的变化，以及投毒占格</caption>' +
      '<thead><tr><th scope="col">题</th><th scope="col">组</th><th scope="col">问题</th><th scope="col">应命中</th>' +
      '<th scope="col">正确答案位次<br><span class="muted small">全池原始排序：关→攻→防</span></th>' +
      '<th scope="col">正确答案位次<br><span class="muted small">top-3 返回列表：关→攻→防</span></th>' +
      '<th scope="col">投毒占格<br><span class="muted small">攻→防</span></th>' +
      '<th scope="col">结论</th></tr></thead><tbody>' + body + '</tbody></table>' +
      '<p class="muted small">口径：hit@3 判的是「top-3 返回列表」那一列（出榜＝没进前 3）。' +
      '「全池原始排序」不截断、也不套同源配额，只是用同一份分数、同一个比较器把候选集整批排一遍，用来看被挤动的幅度；' +
      '两列可以不一致——比如配额把上方同篇的块剔掉后，原始第 4 名就成了返回第 3 名。' +
      '这张表不产生新数据，读的都是页面已有排名。</p></div>');
  }

  /* B 方案：扫描表里 6/10 那类组合级分数，页内只给到组合级，逐题明细不打包进产物。
     复算结论留在这里（也用作作品帖素材），页内只写一行指路说明：
       仅 R2@线上阈值 = 6/10，掉的两题是 Q5（zip：关闭态返回第 3 → 出榜，全池原始 4 → 68）
       与 Q8（rglob：关闭态返回第 1 → 出榜，全池原始 1 → 13），其余 8 题位次不变。
     逐题中间态在构建脚本 tools/build_corpus.py 的 measure() 里，本地跑一次就有；
     本轮不重跑脚本、不改 data.js，所以页内不展示这两题的明细。 */
  function renderScanNote() {
    if (document.getElementById('atkScanNote')) return;
    var tbody = $('atkScanBody');
    var wrap = tbody && tbody.closest ? tbody.closest('.tablewrap') : null;
    if (!wrap) return;
    wrap.insertAdjacentHTML('afterend',
      '<p class="muted small" id="atkScanNote">' +
      '这张表只到组合级：页内没有打包 6 个组合 × ' + PRESETS.length + ' 道题的逐题名次（那要把本页用不到的 5 行中间态一起塞进产物）。' +
      '「仅 R2@线上阈值」那一行 6/10 掉的是哪两题、怎么掉的，明细在构建脚本 tools/build_corpus.py 的 measure() 里，本地跑一次可复算；' +
      '本页采用组合（' + esc(A.defCombo) + '）的逐题位次变化，则列在上方「逐题位次差值」那张表里，可直接对照。</p>');
  }

  function checkAgainst(rows, want) {
    var ok = 0;
    rows.forEach(function (r, k) {
      if (r.srcs.join('|') === want[k].join('|')) ok++;
    });
    return { ok: ok, total: rows.length };
  }

  /* ---------- 渲染：参数 / 说明 / 溯源 ---------- */
  function renderStatic() {
    var pr = DATA.provenance;
    $('topKLabel').textContent = P.TOP_K;
    $('privN').textContent = N;
    var items = [
      ['TOP_K', P.TOP_K, '检索几块喂给模型'],
      ['MAX_PER_SOURCE', P.MAX_PER_SOURCE, '同一篇最多贡献几块'],
      ['KW_WEIGHT', P.KW_WEIGHT, '字面腿权重（0＝排名只看语义）'],
      ['SRC_PENALTY', P.SRC_PENALTY, '块数超中位的笔记字面分打折'],
      ['DROP_Z', P.DROP_Z, '稳健 z 阈值，判疑似词面堆砌'],
      ['DROP_FACTOR', P.DROP_FACTOR, '降权系数；1.0＝降权关闭'],
      ['CHUNK_SIZE', P.CHUNK_SIZE, '每块字数'],
      ['CHUNK_OVERLAP', P.CHUNK_OVERLAP, '相邻块重叠字数'],
    ];
    $('paramsBox').innerHTML = items.map(function (x) {
      return '<div><dt>' + x[0] + '<br><span class="muted small">' + x[2] + '</span></dt><dd>' + x[1] + '</dd></div>';
    }).join('');
    $('paramNote').textContent = '这些取值与本地 obsidian_rag.py 一致，页面只读展示、不提供调参——参数一改，' +
      '打包进来的语义分就不再对应这套排名，看到的分数会失去意义。';

    var cons = consistency();
    var atkCons = consistency('atk'), defCons = consistency('def');
    var evAll = evalSet('all', 'off'), evAOnly = evalSet('A', 'off'), evBOnly = evalSet('B', 'off');
    var notes = [
      '语义分：离线用 ' + pr.model + ' 对每个笔记块与每道预置题实算的余弦值（' + DATA.meta.dims + ' 维，单位模长），' +
      '保留 ' + pr.scorePrecision + ' 位小数；全库相邻名次最小分差 ' + pr.minTopGap.toFixed(6) + '，舍入不改变排序。',
      '字面分：浏览器内实时计算，公式与参考实现一致——BM25 形式 IDF、词频饱和 1+log(count)、min-max 归一，' +
      '再乘 source 宽度惩罚（乘在归一化之后）。',
      '中文分词：预置题使用离线导出的 jieba 真值分词；自由输入降级为「连续汉字 2-gram + 英文标识符正则」，' +
      '所以自由输入的字面分是近似值，且它没有语义分。',
      '评测集溯源：这 ' + evAll.n + ' 道题就是本地 rag_eval.py 里的同一份 EVAL 表（题面与 targets 逐题一致，没有增删），' +
      'A 组 ' + evAOnly.n + ' 道原词命中型（问题里直接带关键词／标识符）＋ B 组 ' + evBOnly.n +
      ' 道语义改写型（问题里【故意不出现】目标笔记的关键词，用来防「字面分压死语义」这种自证）。' +
      '当前打包参数 KW_WEIGHT=' + P.KW_WEIGHT + '（排名只看语义），这一口径下实测 A 组 ' + evAOnly.hit3 + '/' + evAOnly.n +
      '、B 组 ' + evBOnly.hit3 + '/' + evBOnly.n + '，合起来就是页面顶部那个 ' + evAll.hit3 + '/' + evAll.n +
      '——不是 8/8。本地脚本注释里写的 8/8 说的只是 A 组，那份脚本自己打印的分母也是 10 题。',
      'B 组 0/2 是探针，不是调参失误。这两题在本地脚本里就是专门设计来防「字面分太强压死语义」的（rag_eval.py:38-41），' +
      '题面故意不出现目标笔记里的关键词。在当前纯语义口径（KW_WEIGHT=' + P.KW_WEIGHT + '）下两条通道都够不到：' +
      'Q9 目标块最好余弦 0.536251、全库第 38 名，Q10 全库第 6 名；同一目标笔记换个说法，名次从第 4 掉到第 38。' +
      '所以这 0/2 暴露的是语义泛化的边界，hit@3 ' + evAll.hit3 + '/' + evAll.n + ' 不等于「这套系统只有 80 分」，' +
      '它的读法是「A 组原词命中 ' + evAOnly.hit3 + '/' + evAOnly.n + '，B 组改写型 ' +
      evBOnly.hit3 + '/' + evBOnly.n + '」。',
      '为什么不把向量搬进浏览器：' + N + ' 块 × ' + DATA.meta.dims + ' 维即量化成 Int8，噪声约 6.4e-3 已大于实测第 1/2 名分差 5.2e-3，会排错 hit@1；' +
      '改为直接打包实算相似度，体积小 18 倍且是真值。',
      '本页不做答案生成：生成需要大模型，任何现场调用都违反「零外部 API／零 localhost」的约束，' +
      '所以只做检索与评测这一层——也正是「检索怎么命中」这件事本身。',
      '语料脱敏：共替换 ' + Object.keys(pr.scrubbed).length + ' 类身份信息（' + JSON.stringify(pr.scrubbed) + '），' +
      '被改动的 ' + pr.reEmbedded + ' 块已重新向量化，保证文本与向量对应同一份内容。',
      '一致性自检：' + (cons.ok === true ? '通过，浏览器算法与 search() 逐题一致。'
        : cons.ok === false ? '发现不一致：' + cons.bad.map(function (x) { return x.q; }).join('；')
          : '无参考基线可比对。'),
      '零请求：不 fetch、不连本地端口、不加载 CDN；语料与参数在 data.js 内。',

      '开关①「注入攻击演示」：只在候选集上做文章——攻击档把 ' + A.poison.length + ' 个投毒块（' + A.docs.length +
      ' 份样本）并进同一套排名，防御档在排名前按 ' + A.defCombo + ' 剔块，参数与内嵌语料一个都不改，' +
      '关掉开关就是原来的结果。防御组合与阈值 L=' + A.echoLen + ' 由离线扫描按「hit@3 不掉 → 投毒最少 → 误伤最少」三级判据选出，' +
      '不是手工挑的；' + A.defenseScan.length + ' 个组合的实测都列在攻防面板的扫描表里。',
      'R1 / R5 是本演示页新增的规则（参考实现里没有），并且都拦不住改写后的载荷——' +
      '攻防面板里 P4（改写绕过）与 P5（不抄提问的静默指令）就是专门用来暴露这两条天花板，' +
      'P3（关键词堆砌）留在那里没被剔，是因为唯一能拦它的 R2 在真实库上会连正常长文一起打（本库实测代价 ' +
      (A.defenseScan.filter(function (s) { return s.name.indexOf('仅 R2') >= 0; })[0] || { hit3: '—' }).hit3 +
      '/10）。这与你笔记里的结论一致：靠字面过滤做防御本质上拦不住。',
      '攻防两档复核的是排名引擎：浏览器排出的名次序列与参考实现 search() 在同一候选集上逐题一致（投毒态 ' +
      (atkCons.total - atkCons.bad.length) + '/' + atkCons.total + ' 题、防御态 ' +
      (defCons.total - defCons.bad.length) + '/' + defCons.total + ' 题，攻防面板底部那行自检就是这个数）；' +
      '但 R1 / R5 两条规则本身为本产物实现，参考实现里没有——所以上面那张扫描表里各组合的 hit@3、投毒进榜、误伤三列，' +
      '读作「在这两条规则可检出的条件下、于同一份候选集实测出来的结果」，不是与外部实现对照过的结论。',
      '开关②「换我的语料」：粘贴的文本只留在当前页面的内存变量里——不写 localStorage、不发往任何服务器、' +
      '不进评测集，刷新即回到内嵌的 ' + N + ' 块。这条通道没有语义分（静态页跑不了 embedding），' +
      '只有字面分，用来看「清洗与分块怎么切、词怎么命中」。',
      '分块保真自检：打包时把 3 段样本的参考实现输出（clean() → chunk()）一起存进产物，' +
      '页面每次加载都用同一份 JS 重跑一遍并逐块比对，' + (SELF_CHUNK.ok ? '当前 ' + SELF_CHUNK.ok + '/' + SELF_CHUNK.total + ' 段逐块一致' : '当前不一致：' + SELF_CHUNK.bad) +
      '——保证你看到的切块就是本地那套规则切出来的。',
    ];
    $('notesBox').innerHTML = notes.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('');

    /* ---- B 组探针 rank 对比（上一轮「可选加分项」落地） ----
       只读 rawRank()——攻防面板逐题位次表已在用的同一个函数，不新增数据、不重跑构建、不改任何排名结果。
       口径：这里展示「正确答案在全池排第几」的出榜名次（不截断、不套同源配额），与 hit@3 的 top-3
       命中口径不同，二者不可互替阅读——面板顶部固定写明这一点（对应上轮记录的风险①）。
       Q9 / Q10 各自独立成行并标注目标笔记，不并成一句（风险②）；本面板位于评测明细区，
       不与攻防面板的八列位次表重复（风险③）。默认展开，供现场核数。 */
    (function attachProbeCompare(){
      var notesEl = $('notesBox');
      var li = null;
      for (var k = 0; k < notesEl.children.length; k++) {
        if (notesEl.children[k].textContent.indexOf('B 组 0/2 是探针') === 0) { li = notesEl.children[k]; break; }
      }
      if (!li) return;
      var bIds = setPresets('B');
      if (!bIds.length) return;
      var box = document.createElement('div');
      box.className = 'probe-compare';
      var head = document.createElement('button');
      head.type = 'button';
      head.className = 'pc-toggle';
      head.textContent = '▸ 收起 B 组 rank 对比（全池位次 · 页面现场实算）';
      var detail = document.createElement('div');
      detail.className = 'pc-detail';
      var html = '<p class="pc-note">下面是「正确答案在全池 ' + N + ' 个块里排第几」的出榜名次——不截断、不套同源配额，' +
        '与上方 hit@3 的 top-3 口径不同，不能互相替代阅读。数字由页面现场调用攻防面板同款的只读位次函数实算，' +
        '不是写死在产物里的。</p>';
      bIds.forEach(function (i) {
        var p = PRESETS[i], rb = rawRank('off', i), aJ = -1;
        setPresets('A').forEach(function (j) {
          if (aJ >= 0) return;
          p.targets.forEach(function (t) { if (PRESETS[j].targets.indexOf(t) >= 0) aJ = j; });
        });
        html += '<div class="pc-row"><div class="pc-q">Q' + (i + 1) + ' 「' + esc(p.query) + '」' +
          '<span class="pc-tgt">目标笔记：' + esc(p.targets[0]) + '</span></div>' +
          '<div class="pc-bars">' +
          (aJ >= 0
            ? '<span class="pc-cell"><i>原词问法（A 组 Q' + (aJ + 1) + '）</i><b>#' + rawRank('off', aJ) + '</b></span>' +
              '<span class="pc-arrow">→</span>'
            : '<span class="pc-cell na"><i>本组无同目标原词题</i><b>—</b></span>') +
          '<span class="pc-cell bad"><i>改写问法（本题）</i><b>#' + rb + '</b></span>' +
          '</div></div>';
      });
      detail.innerHTML = html;
      head.addEventListener('click', function () {
        var open = detail.style.display !== 'none';
        detail.style.display = open ? 'none' : '';
        head.textContent = open
          ? '▸ 展开 B 组 rank 对比（全池位次 · 页面现场实算）'
          : '▸ 收起 B 组 rank 对比（全池位次 · 页面现场实算）';
      });
      box.appendChild(head);
      box.appendChild(detail);
      li.appendChild(box);
    })();

    var prov = [
      ['生成来源', 'obsidian_rag.py + rag_eval.py 同数据'],
      ['embedding 模型', pr.model],
      ['块数 / 篇数', pr.indexChunks + ' / ' + DATA.meta.sources],
      ['chunk_size / overlap', DATA.meta.chunk_size + ' / ' + DATA.meta.overlap],
      ['索引版本', 'v' + DATA.meta.version + ' · pipeline v' + DATA.meta.pipeline_version],
    ];
    $('provBox').innerHTML = prov.map(function (x) {
      return '<div><dt>' + x[0] + '</dt><dd>' + esc(String(x[1])) + '</dd></div>';
    }).join('');
  }

  /* ---------- 会话操作记录（仅本会话，非数据源） ---------- */
  var LOG_PREFIX = 'ragbench.';
  var LOG_KEY = LOG_PREFIX + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
  var memLog = [];
  var storage = true;
  try {
    Object.keys(localStorage).forEach(function (k) {
      if (k.indexOf(LOG_PREFIX) === 0) localStorage.removeItem(k);   // 只留本次会话
    });
    localStorage.setItem(LOG_KEY, '[]');
    localStorage.removeItem(LOG_KEY);
  } catch (e) { storage = false; }

  function fmtTime(d) {
    return d.toLocaleTimeString('zh-CN', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  function log(action, detail) {
    memLog.push({ t: fmtTime(new Date()), a: action, d: detail == null ? '' : String(detail) });
    if (storage) {
      try { localStorage.setItem(LOG_KEY, JSON.stringify(memLog)); } catch (e) { storage = false; }
    }
    renderLog();
  }
  function renderLog() {
    $('logList').innerHTML = memLog.length
      ? memLog.slice().reverse().map(function (r) {
        return '<li><span class="t">' + r.t + '</span><span class="a">' + esc(r.a) + '</span><span>' + esc(r.d) + '</span></li>';
      }).join('')
      : '<li><span class="t">—</span><span class="a">空</span><span>本会话还没有操作</span></li>';
    if (!storage && !$('logWarn')) {
      $('logList').insertAdjacentHTML('afterend',
        '<p class="muted small" id="logWarn">浏览器不允许本地存储（无痕或配额已满），记录只在内存里保留，页面功能不受影响。</p>');
    }
  }
  $('clearLog') && $('clearLog').addEventListener('click', function () {
    memLog = [];
    try { localStorage.removeItem(LOG_KEY); } catch (e) { }
    log('清空记录', '之后的操作会重新写入');
  });

  /* ---------- 对比度自适应 ---------- */
  function parseColor(str) {
    if (!str) return null;
    var d = document.createElement('span');
    d.style.color = str;
    d.style.display = 'none';
    document.body.appendChild(d);
    var got = getComputedStyle(d).color;
    document.body.removeChild(d);
    var m = got.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    var parts = m[1].split(/[,\s/]+/).filter(Boolean).map(Number);
    return [parts[0], parts[1], parts[2]];
  }
  function lum(rgb) {
    var a = rgb.map(function (v) {
      v /= 255;
      return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
  }
  function contrast(a, b) {
    var l1 = lum(a), l2 = lum(b);
    return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
  }
  function toHsl(rgb) {
    var r = rgb[0] / 255, g = rgb[1] / 255, b = rgb[2] / 255;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b), h = 0, s = 0, l = (mx + mn) / 2;
    if (mx !== mn) {
      var d = mx - mn;
      s = l > 0.5 ? d / (2 - mx - mn) : d / (mx + mn);
      h = mx === r ? (g - b) / d + (g < b ? 6 : 0) : mx === g ? (b - r) / d + 2 : (r - g) / d + 4;
      h /= 6;
    }
    return [h * 360, s * 100, l * 100];
  }
  function fromHsl(hsl) {
    var h = hsl[0] / 360, s = hsl[1] / 100, l = hsl[2] / 100;
    function f(p, q, t) {
      if (t < 0) t += 1; if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    }
    var r, g, b;
    if (s === 0) { r = g = b = l; } else {
      var q = l < 0.5 ? l * (1 + s) : l + s - l * s, p = 2 * l - q;
      r = f(p, q, h + 1 / 3); g = f(p, q, h); b = f(p, q, h - 1 / 3);
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }
  function fit(varName, bg, target) {
    var rootCS = getComputedStyle(document.documentElement);
    var raw = rootCS.getPropertyValue(varName).trim();
    var rgb = parseColor(raw) || parseColor(rootCS.color);
    if (!rgb) return;
    if (contrast(rgb, bg) >= target) {
      document.documentElement.style.setProperty(
        varName === '--site-fg' ? '--fg-adj' : '--muted-adj',
        'rgb(' + rgb.join(',') + ')');
      return;
    }
    var hsl = toHsl(rgb);
    var lighter = lum(bg) < 0.5;
    var best = rgb;
    for (var step = 0; step <= 100; step++) {
      var l = lighter ? Math.min(100, hsl[2] + step) : Math.max(0, hsl[2] - step);
      var cand = fromHsl([hsl[0], hsl[1], l]);
      if (contrast(cand, bg) >= target) { best = cand; break; }
    }
    document.documentElement.style.setProperty(
      varName === '--site-fg' ? '--fg-adj' : '--muted-adj',
      'rgb(' + best.join(',') + ')');
  }
  function adaptContrast() {
    var cs = getComputedStyle(document.documentElement);
    var bg = parseColor(cs.getPropertyValue('--site-bg').trim()) || [245, 247, 249];
    fit('--site-fg', bg, 4.5);
    fit('--site-muted', bg, 3.2);
  }

  /* ---------- 事件：提问区 ---------- */
  $('askForm').addEventListener('submit', function (e) {
    e.preventDefault();
    var v = $('q').value.trim();
    if (!v) { $('q').focus(); return; }
    if (state.corpus === 'mine') {
      if (!CUSTOM.applied) { $('corpusInput').focus(); return; }
      state.customQ = v;                       // 自定义语料：只走字面检索，不进评测集
      log('自由提问（自定义语料）', v);
    } else {
      state.mode = 'free';
      state.query = v;
      log('自由提问', v);
    }
    renderAll();
  });
  $('resetBtn').addEventListener('click', function () {
    $('q').value = '';
    if (state.corpus === 'mine') { state.customQ = ''; renderAll(); return; }
    state.mode = 'preset';
    state.presetIdx = setPresets(state.set)[0] || 0;
    log('清空输入', '回到预置题 #' + (state.presetIdx + 1));
    renderAll();
  });
  $('runEval').addEventListener('click', function () {
    if (state.corpus === 'mine') { $('runState').textContent = '自定义语料不参与评测（没有语义向量）'; return; }
    state.ranEval = true;
    log('跑一遍评测', '评测集=' + ({ all: '全部', A: 'A 组', B: 'B 组' })[state.set]);
    renderScore();
    renderEval();
    $('evalPanel').scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
  });
  Array.prototype.forEach.call(document.querySelectorAll('.segb[data-set]'), function (b) {
    b.addEventListener('click', function () {
      Array.prototype.forEach.call(document.querySelectorAll('.segb[data-set]'), function (x) {
        x.classList.remove('on'); x.setAttribute('aria-pressed', 'false');
      });
      b.classList.add('on'); b.setAttribute('aria-pressed', 'true');
      state.set = b.getAttribute('data-set');
      state.presetIdx = setPresets(state.set)[0] || 0;
      state.mode = 'preset';
      log('切换评测集', b.textContent.trim());
      renderAll();
      if (state.ranEval) renderEval();
    });
  });

  /* ---------- 事件：开关①「注入攻击演示」 ---------- */
  var ATK_LABEL = { off: '关', atk: '攻击', def: '攻击 + 防御' };
  var LAST_ATK = null;   // 切到自定义语料时暂存的攻击档位，切回内嵌语料自动还原
  function setAtkMode(m, doLog) {
    state.atk = m;
    Array.prototype.forEach.call(document.querySelectorAll('.atkb'), function (x) {
      var on = x.getAttribute('data-atk') === m;
      x.classList.toggle('on', on);
      x.classList.toggle('bad-on', on && m === 'atk');
      x.classList.toggle('ok-on', on && m === 'def');
      x.setAttribute('aria-pressed', String(on));
    });
    $('atkPanel').hidden = m === 'off';
    if (doLog) {
      log('注入攻击演示', ATK_LABEL[m] + (m === 'off' ? ''
        : '（防御组合 ' + A.defCombo + '，复读阈值 L=' + A.echoLen + '）'));
    }
  }
  Array.prototype.forEach.call(document.querySelectorAll('.atkb'), function (b) {
    b.addEventListener('click', function () {
      var m = b.getAttribute('data-atk');
      if (m === state.atk) return;
      setAtkMode(m, true);
      renderAll();
      if (m !== 'off') {
        renderAttackPanel();
        $('atkPanel').scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
      }
    });
  });

  /* ---------- 事件：开关②「换我的语料」 ---------- */
  Array.prototype.forEach.call(document.querySelectorAll('.corpb'), function (b) {
    b.addEventListener('click', function () {
      var c = b.getAttribute('data-corpus');
      if (c === state.corpus) return;
      if (c === 'mine' && !CUSTOM.applied && !CUSTOM.raw.trim()) {
        state.corpus = 'mine';
        $('corpusPanel').hidden = false;
        syncCorpusSeg(c);
        setAtkEnabled(false, '先在上面贴一段文本并点「用这段检索」');
        log('换我的语料', '打开粘贴框（内嵌语料仍在，评测口径不变）');
        renderAll();
        $('corpusPanel').scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
        $('corpusInput').focus();
        return;
      }
      state.corpus = c;
      if (c === 'builtin') {
        $('q').placeholder = '例如：字符串不可变，怎么改其中一个字符';
        setAtkEnabled(true, ''); syncCorpusSeg(c); $('corpusPanel').hidden = true;
        renderCorpusPanel();   // 面板虽然收起，也别把上一次的切块结果留在 DOM 里
      }
      else { syncCorpusSeg(c); $('corpusPanel').hidden = false; setAtkEnabled(false, '自定义语料只做字面检索'); }
      log('语料来源', c === 'builtin' ? '内嵌语料（' + N + ' 块）' : '我粘贴的文本（' + CUSTOM.texts.length + ' 块，仅本页面内存）');
      renderAll();
    });
  });
  function syncCorpusSeg(cur) {
    Array.prototype.forEach.call(document.querySelectorAll('.corpb'), function (x) {
      var on = x.getAttribute('data-corpus') === cur;
      x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on));
    });
  }
  function setAtkEnabled(on, why) {
    Array.prototype.forEach.call(document.querySelectorAll('.atkb'), function (x) {
      x.disabled = !on;
      x.title = on ? '' : ('注入攻击演示需要内嵌语料：' + why);
    });
    if (!on && state.atk !== 'off') {
      LAST_ATK = state.atk;                 // 记住访客选到哪一档，回到内嵌语料时还原
      setAtkMode('off', false);
    } else if (on && LAST_ATK && LAST_ATK !== 'off') {
      var m = LAST_ATK; LAST_ATK = null;
      setAtkMode(m, false);
      log('注入攻击演示', '回到内嵌语料，恢复你之前选的档位：' + ATK_LABEL[m]);
    }
  }
  $('corpusInput').addEventListener('input', function () {
    CUSTOM.raw = $('corpusInput').value;
    $('corpusCharN').textContent = CUSTOM.raw.length + ' / 4000 字';
    renderCorpusPanel();                       // 只预览分块，未点「用这段检索」前不替换语料
  });
  $('corpusApply').addEventListener('click', function () {
    if (!CUSTOM.raw.trim()) { $('corpusInput').focus(); $('corpusState').textContent = '先粘贴一段文本'; return; }
    CUSTOM.texts = jsChunk(jsClean(CUSTOM.raw), P.CHUNK_SIZE, P.CHUNK_OVERLAP);
    CUSTOM.applied = CUSTOM.texts.length > 0;
    if (!CUSTOM.applied) {
      $('corpusState').textContent = '清洗后没有留下文字，切不出块——贴点正文试试';
      $('corpusInput').focus();
      return;
    }
    state.corpus = 'mine';
    state.customQ = '';
    $('q').value = '';
    $('q').placeholder = '用你刚贴的文本里出现过的词提问（这条通道只有字面分）';
    $('corpusPanel').hidden = false; syncCorpusSeg('mine'); setAtkEnabled(false, '自定义语料只做字面检索');
    log('换我的语料', '已用粘贴文本替换临时语料：清洗后 ' + CUSTOM.texts.length + ' 块（不进评测集、不写存储）');
    renderAll();
    $('q').focus();
  });
  $('corpusReset').addEventListener('click', function () {
    CUSTOM.raw = ''; CUSTOM.texts = []; CUSTOM.applied = false; state.customQ = '';
    $('corpusInput').value = ''; $('corpusCharN').textContent = '0 / 4000 字';
    $('q').value = ''; $('q').placeholder = '例如：字符串不可变，怎么改其中一个字符';
    state.corpus = 'builtin';
    $('corpusPanel').hidden = true; syncCorpusSeg('builtin'); setAtkEnabled(true, '');
    log('恢复内嵌语料', N + ' 块 / ' + DATA.meta.sources + ' 篇笔记；粘贴文本已从内存清除');
    renderAll();
    renderCorpusPanel();          // 收起的面板里也别留下上一次的切块预览
  });

  function renderAll() {
    renderScore();
    renderPresets();
    renderResults();
    if (state.ranEval) renderEval();
    if (!$('atkPanel').hidden) renderAttackPanel();
    if (!$('corpusPanel').hidden) renderCorpusPanel();
  }

  /* ---------- 启动 ---------- */
  adaptContrast();
  renderStatic();
  renderAll();
  log('打开页面', N + ' 块 / ' + DATA.meta.sources + ' 篇笔记已就绪（攻防两个开关默认关闭）');

  try {
    var mq = matchMedia('(prefers-color-scheme: dark)');
    (mq.addEventListener ? mq.addEventListener('change', adaptContrast) : mq.addListener(adaptContrast));
    new MutationObserver(adaptContrast).observe(document.documentElement, { attributes: true, attributeFilter: ['style', 'class'] });
    new MutationObserver(adaptContrast).observe(document.head, { childList: true });
  } catch (e) { /* 观察器不可用时保持初始配色 */ }

  window.addEventListener('error', function (ev) {
    // 任何意外都不让页面变成白屏：记录一次，界面继续可用
    try { log('运行时提示', (ev && ev.message) || '未知错误'); } catch (e) { }
  });
})();
