// Screenshot the first section of every interactive page, for previews in the write-ups.
// 为每个交互页的第一节截图，作为讲解中的预览图。
// Usage / 用法: node scripts/screenshot_previews.js   (needs `playwright` installed)
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const ROOT = path.resolve(__dirname, "..");

(async () => {
  const browser = await chromium.launch();
  const docs = path.join(ROOT, "docs");
  for (const topic of fs.readdirSync(docs)) {
    const dir = path.join(docs, topic, "interactive");
    if (!fs.existsSync(dir)) continue;
    const pages = fs.readdirSync(dir).filter(f => f.endsWith(".html")).sort();
    // A chapter's first page is its main playground; every Q&A page gets its own preview
    // 每章第一个页面作为主演示；问答单元的每个页面各有一张预览
    const shots = topic === "qa"
      ? pages.map(f => [f, `${path.basename(f, ".html")}-preview`])
      : pages.slice(0, 1).map(f => [f, "interactive-preview"]);
    for (const [name, stem] of shots) for (const [lang, suffix] of [["zh", ""], ["en", ".en"]]) {
      const file = path.join(dir, name);
      const page = await browser.newPage({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 1 });
      await page.goto(`file://${file}?lang=${lang}`);
      const out = path.join(docs, topic, "assets", `${stem}${suffix}.png`);
      const top = await page.locator("main").boundingBox();
      const first = await page.locator("section").first().boundingBox();
      await page.screenshot({
        path: out,
        clip: { x: top.x, y: top.y, width: top.width, height: first.y + first.height - top.y + 16 },
      });
      console.log(`saved ${path.relative(ROOT, out)}`);
      await page.close();
    }
  }
  await browser.close();
})();
