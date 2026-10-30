"""Multimodal models in PyTorch / 用 PyTorch 实现多模态模型.

Run / 运行:
    python docs/16-multimodal/code/multimodal.py

Covers / 包含:
    1. a tiny world of pictures and captions: coloured shapes, drawn from code
       一个由代码绘制的小世界：彩色图形及其描述
    2. a Vision Transformer (ViT): cut the image into patches, one token per patch; the patch embedding
       is exactly a strided convolution
       视觉 Transformer（ViT）：把图像切成小块，每块一个词元；块嵌入就是步长等于块大小的卷积
    3. CLIP: train an image encoder and a text encoder so matching pairs point the same way (a symmetric
       contrastive loss), then classify images zero-shot with text prompts
       CLIP：用对称的对比损失训练图像编码器和文本编码器，让匹配的图文指向同一方向，再用文本提示做零样本分类
    4. a LLaVA-style vision-language model: frozen image encoder -> projector -> a small language model
       that writes the caption
       LLaVA 式视觉语言模型：冻结的图像编码器 -> 投影层 -> 写描述的小语言模型
    5. what they get wrong: colour-shape combinations held out of training
       它们会错在哪里：训练中留出的颜色-形状组合
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. A tiny world of pictures and captions / 图与描述的小世界
# ---------------------------------------------------------------------------
SIZE = 16
COLORS = {"red": (0.9, 0.15, 0.15), "green": (0.15, 0.75, 0.2), "blue": (0.2, 0.3, 0.95), "yellow": (0.95, 0.85, 0.1)}
SHAPES = ["circle", "square", "triangle", "cross"]
SIZES = ["small", "big"]
SIDES = ["left", "right"]
HELD_OUT = {("green", "triangle"), ("yellow", "circle"), ("blue", "cross"), ("red", "square")}  # never seen in training
WORDS = ["<pad>", "<bos>", "<eos>", "a", "on", "the"] + list(COLORS) + SHAPES + SIZES + SIDES
STOI = {w: i for i, w in enumerate(WORDS)}
_yy, _xx = torch.meshgrid(torch.arange(SIZE).float(), torch.arange(SIZE).float(), indexing="ij")


def draw(color, shape, size, side, rng):
    """A 3 x 16 x 16 image with one shape on a dark, slightly noisy background / 深色略带噪声的背景上画一个图形."""
    r = (2.2 if size == "small" else 3.6) + rng.uniform(-0.3, 0.3)
    cx = (4.5 if side == "left" else 11.5) + rng.uniform(-1, 1)
    cy = 8 + rng.uniform(-2.5, 2.5)
    dx, dy = _xx - cx, _yy - cy
    if shape == "circle":
        mask = dx ** 2 + dy ** 2 <= r ** 2
    elif shape == "square":
        mask = (dx.abs() <= r * 0.85) & (dy.abs() <= r * 0.85)
    elif shape == "triangle":
        mask = (dy <= r * 0.8) & (dy >= -r) & (dx.abs() <= (dy + r) * 0.55)
    else:
        mask = ((dx.abs() <= r * 0.3) & (dy.abs() <= r)) | ((dy.abs() <= r * 0.3) & (dx.abs() <= r))
    noise = torch.rand(3, SIZE, SIZE, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))
    color_rgb = torch.tensor(COLORS[color])[:, None, None] + rng.uniform(-0.08, 0.08)
    return torch.where(mask[None], color_rgb, 0.08 + 0.05 * noise).clamp(0, 1)


def sample(n, rng, held_out=False):
    """n images with captions like "a big red circle on the left"; held_out picks only the held-out
    colour-shape pairs. 生成 n 张图及描述；held_out=True 时只取留出的颜色-形状组合。"""
    images, captions = [], []
    for _ in range(n):
        while True:
            color, shape = rng.choice(list(COLORS)), rng.choice(SHAPES)
            if ((color, shape) in HELD_OUT) == held_out:
                break
        size, side = rng.choice(SIZES), rng.choice(SIDES)
        images.append(draw(color, shape, size, side, rng))
        captions.append(["a", size, color, shape, "on", "the", side])
    return torch.stack(images), captions


def encode(captions):
    return torch.tensor([[STOI[w] for w in c] for c in captions])


# ---------------------------------------------------------------------------
# 2. Vision Transformer / 视觉 Transformer
# ---------------------------------------------------------------------------
def patchify(images, p):
    """(B, 3, H, W) -> (B, H/p * W/p, 3 * p * p): every p x p patch flattened into one vector.
    把每个 p x p 的小块展平成一个向量。"""
    return F.unfold(images, kernel_size=p, stride=p).transpose(1, 2)


def encoder(d, layers):
    layer = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ViT(nn.Module):
    """Patches -> linear embedding (a strided conv) + positions + a [CLS] token -> Transformer.
    小块 -> 线性嵌入（步长卷积）+ 位置 + [CLS] 词元 -> Transformer。"""

    def __init__(self, d=64, patch=4, layers=2):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch, stride=patch)
        n = (SIZE // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.randn(1, n + 1, d) * 0.02)
        self.blocks = encoder(d, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, all_tokens=False):
        t = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, patches, d)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], dim=1) + self.pos
        h = self.norm(self.blocks(t))
        return h if all_tokens else h[:, 0]


class TextEncoder(nn.Module):
    def __init__(self, d=64, layers=2):
        super().__init__()
        self.embed = nn.Embedding(len(WORDS), d)
        self.pos = nn.Parameter(torch.randn(1, 8, d) * 0.02)
        self.blocks = encoder(d, layers)
        self.norm = nn.LayerNorm(d)

    def forward(self, ids):
        return self.norm(self.blocks(self.embed(ids) + self.pos[:, :ids.size(1)])).mean(1)


# ---------------------------------------------------------------------------
# 3. CLIP / 图文对比学习
# ---------------------------------------------------------------------------
class CLIP(nn.Module):
    def __init__(self, d=64, dim=32):
        super().__init__()
        self.vision, self.text = ViT(d), TextEncoder(d)
        self.proj_image, self.proj_text = nn.Linear(d, dim, bias=False), nn.Linear(d, dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))  # learned temperature / 可学习的温度

    def embed_image(self, x):
        return F.normalize(self.proj_image(self.vision(x)), dim=-1)

    def embed_text(self, ids):
        return F.normalize(self.proj_text(self.text(ids)), dim=-1)


def clip_loss(img, txt, scale):
    """Symmetric cross-entropy over the batch's similarity matrix: image i should pick caption i among
    all captions, and caption i should pick image i. 批内相似度矩阵上的对称交叉熵：第 i 张图要在所有描述中选中
    第 i 条，第 i 条描述也要选中第 i 张图。"""
    logits = scale * img @ txt.T
    labels = torch.arange(len(img))
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_clip(steps, batch=128, seed=0, log=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CLIP()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    for step in range(steps):
        x, caps = sample(batch, rng)
        loss = clip_loss(model.embed_image(x), model.embed_text(encode(caps)), model.logit_scale.exp())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log is not None and step % 25 == 0:
            log.append((step, loss.item()))
    model.eval()
    return model


PAIRS = [(c, s) for c in COLORS for s in SHAPES]


@torch.no_grad()
def prompt_embeddings(model, ensemble=True):
    """One text embedding per colour-shape pair, from one template or averaged over all of them.
    每个颜色-形状组合一个文本向量：只用一个模板，或者对所有模板取平均（提示词集成）。"""
    templates = [(z, d) for z in SIZES for d in SIDES] if ensemble else [("small", "left")]
    embs = torch.stack([model.embed_text(encode([["a", z, c, s, "on", "the", d] for c, s in PAIRS])) for z, d in templates])
    return F.normalize(embs.mean(0), dim=-1)


@torch.no_grad()
def zero_shot(model, images, captions, ensemble=True):
    """Classify every image as the colour-shape pair whose prompt is most similar.
    把每张图归为与之最相似的提示所对应的颜色-形状组合。"""
    pred = (model.embed_image(images) @ prompt_embeddings(model, ensemble).T).argmax(-1).tolist()
    n = len(captions)
    return dict(pair=sum(PAIRS[p] == (c[2], c[3]) for p, c in zip(pred, captions)) / n,
                color=sum(PAIRS[p][0] == c[2] for p, c in zip(pred, captions)) / n,
                shape=sum(PAIRS[p][1] == c[3] for p, c in zip(pred, captions)) / n, pred=pred)


def linear_probe(model, steps=300):
    """Train a linear classifier for shape on frozen image features of *seen* pairs, test on held-out
    pairs: is the shape represented separately from the colour? 只用见过的组合在冻结的图像特征上训练形状分类器，
    在留出组合上测试：形状是否独立于颜色被表示？"""
    with torch.no_grad():
        xs, cs = sample(2000, random.Random(11))
        xh, ch = sample(400, random.Random(12), held_out=True)
        fs, fh = model.vision(xs), model.vision(xh)
    ys = torch.tensor([SHAPES.index(c[3]) for c in cs])
    yh = torch.tensor([SHAPES.index(c[3]) for c in ch])
    probe = nn.Linear(fs.size(1), len(SHAPES))
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(steps):
        loss = F.cross_entropy(probe(fs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (probe(fs).argmax(-1) == ys).float().mean().item(), (probe(fh).argmax(-1) == yh).float().mean().item()


# ---------------------------------------------------------------------------
# 4. A LLaVA-style vision-language model / LLaVA 式视觉语言模型
# ---------------------------------------------------------------------------
class VLM(nn.Module):
    """Image patch features (frozen) -> projector -> prefix tokens for a small causal language model.
    图像块特征（冻结）-> 投影层 -> 作为小型因果语言模型的前缀词元。"""

    def __init__(self, vision, d=64):
        super().__init__()
        self.vision = vision
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.proj = nn.Sequential(nn.Linear(vision.norm.normalized_shape[0], d), nn.GELU(), nn.Linear(d, d))
        self.embed = nn.Embedding(len(WORDS), d)
        self.pos = nn.Parameter(torch.randn(1, 32, d) * 0.02)
        self.blocks = encoder(d, 2)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(WORDS))

    def forward(self, images, ids):
        with torch.no_grad():
            v = self.vision(images, all_tokens=True)          # (B, 1 + patches, d)
        h = torch.cat([self.proj(v), self.embed(ids)], dim=1)  # image tokens, then text / 先图像词元，再文字
        T = h.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(T)
        h = self.norm(self.blocks(h + self.pos[:, :T], mask=mask, is_causal=True))
        return self.head(h)[:, v.size(1):]                     # predictions at the text positions / 文字位置的预测

    @torch.no_grad()
    def caption(self, images, length=7):
        ids = torch.full((len(images), 1), STOI["<bos>"])
        for _ in range(length):
            ids = torch.cat([ids, self(images, ids)[:, -1].argmax(-1, keepdim=True)], dim=1)
        return [[WORDS[i] for i in row[1:]] for row in ids.tolist()]


def train_vlm(clip_model, steps, batch=64, seed=0):
    torch.manual_seed(seed)
    rng = random.Random(seed + 100)
    vlm = VLM(clip_model.vision)
    opt = torch.optim.AdamW([p for p in vlm.parameters() if p.requires_grad], lr=2e-3)
    for _ in range(steps):
        x, caps = sample(batch, rng)
        ids, target = encode([["<bos>"] + c for c in caps]), encode([c + ["<eos>"] for c in caps])
        loss = F.cross_entropy(vlm(x, ids).flatten(0, 1), target.flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    vlm.eval()
    return vlm


def caption_scores(out, captions):
    n = len(captions)
    return dict(exact=sum(o == c for o, c in zip(out, captions)) / n,
                color=sum(o[2] == c[2] for o, c in zip(out, captions)) / n,
                shape=sum(o[3] == c[3] for o, c in zip(out, captions)) / n)


# ---------------------------------------------------------------------------
# 5. How many tokens is an image? / 一张图是多少个词元
# ---------------------------------------------------------------------------
def image_tokens(height, width, patch=14, merge=1):
    """Patches per image, optionally merging merge x merge neighbours into one token.
    每张图的块数；可以把相邻的 merge x merge 个块合并成一个词元。"""
    return math.ceil(height / patch / merge) * math.ceil(width / patch / merge)


def main():
    torch.manual_seed(0)
    print("== 1. patches are tokens / 小块就是词元 ==")
    x, caps = sample(4, random.Random(0))
    vit = ViT()
    patches = patchify(x, 4)                                      # (4, 16, 48)
    w = vit.patch_embed.weight.flatten(1)                        # (64, 48)
    linear = patches @ w.T + vit.patch_embed.bias
    conv = vit.patch_embed(x).flatten(2).transpose(1, 2)
    assert patches.shape == (4, 16, 48) and torch.allclose(linear, conv, atol=1e-5)
    print(f"a 16x16 image -> {patches.shape[1]} patches of 4x4x3 = {patches.shape[2]} numbers; "
          f"the patch embedding is a stride-4 convolution  ✓")
    print(f"e.g. {' '.join(caps[0])}")

    print("\n== 2. CLIP: contrastive image-text training / 图文对比训练 ==")
    model = train_clip(steps=500)
    seen_x, seen_c = sample(400, random.Random(5))
    held_x, held_c = sample(400, random.Random(6), held_out=True)
    for ens in [False, True]:
        s, h = zero_shot(model, seen_x, seen_c, ens), zero_shot(model, held_x, held_c, ens)
        print(f"  zero-shot, {'4 templates averaged' if ens else 'one template       '}: seen pairs {s['pair']:.0%}"
              f" (colour {s['color']:.0%}, shape {s['shape']:.0%});  held-out pairs {h['pair']:.0%}"
              f" (colour {h['color']:.0%}, shape {h['shape']:.0%})")
    s, h = zero_shot(model, seen_x, seen_c), zero_shot(model, held_x, held_c)
    assert s["color"] > 0.95 and s["pair"] > 0.55  # learned from captions alone / 只靠描述学会了
    assert h["color"] > 0.9 and h["shape"] < 0.5   # but unseen combinations fail on shape / 没见过的组合，形状出错
    probe_seen, probe_held = linear_probe(model)
    print(f"  linear probe for shape on frozen image features: seen pairs {probe_seen:.0%}, held-out pairs {probe_held:.0%}")
    print("  -> the image features tie shape to colour; unseen combinations look like a seen one  ✓")

    print("\n== 3. a LLaVA-style VLM / LLaVA 式视觉语言模型 ==")
    vlm = train_vlm(model, steps=300)
    out_s, out_h = vlm.caption(seen_x[:200]), vlm.caption(held_x[:200])
    sc_s, sc_h = caption_scores(out_s, seen_c[:200]), caption_scores(out_h, held_c[:200])
    print(f"  seen pairs: {sc_s['exact']:.0%} captions exactly right;  held-out pairs: colour {sc_h['color']:.0%}, shape {sc_h['shape']:.0%}")
    print(f"  e.g. truth '{' '.join(held_c[0])}'  ->  model '{' '.join(out_h[0])}'")
    assert sc_s["exact"] > 0.6 and sc_h["color"] > 0.9 and sc_h["shape"] < 0.3
    print("  it says what it has seen before, not what is there: a hallucination  ✓")

    print("\n== 4. how many tokens is an image? / 一张图是多少个词元 ==")
    for h_, w_, p, m in [(224, 224, 14, 1), (336, 336, 14, 1), (448, 448, 14, 2), (1024, 1024, 16, 1)]:
        print(f"  {h_}x{w_}, patch {p}{', 2x2 merged' if m > 1 else ''}: {image_tokens(h_, w_, p, m):,} tokens")
    assert image_tokens(336, 336, 14) == 576  # LLaVA-1.5's 576 image tokens / LLaVA-1.5 的 576 个图像词元


if __name__ == "__main__":
    main()
