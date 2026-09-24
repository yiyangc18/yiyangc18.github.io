# CHER-YOUNG BLOG

`yiyangc18.github.io` 的 Hexo 源码工程，使用 Fluid 主题。需要 Node.js 20.19 或更高版本。

## 安装依赖

首次克隆后，在项目根目录执行：

```bash
npm install
```

依赖会下载到 `node_modules/`。该目录已在 `.gitignore` 中忽略，不会提交到 Git。`package-lock.json` 需要提交，用于锁定依赖版本。

## 本地预览

```bash
npm run serve
```

打开 <http://localhost:4000/>。修改 Markdown 后页面会自动更新；按 `Ctrl+C` 停止预览。

## 新增文章

```bash
npm run new -- "文章标题"
```

新文章会生成在 `source/_posts/`。可以参考：

```yaml
---
title: 文章标题
date: 2026-09-24 14:00:00
tags:
  - NPU Infra
index_img: /img/example-cover.png
banner_img: /img/tree.png
---
```

### 文章图片

把图片放到 `source/img/`，建议使用有含义的英文文件名，例如：

```text
source/img/npu-runtime-overview.png
```

在 Markdown 中使用从站点根目录开始的路径：

```markdown
![NPU Runtime 架构](/img/npu-runtime-overview.png)
```

## 构建与发布

发布前可在本地构建一次：

```bash
npm run clean
npm run build
```

生成结果位于 `public/`。`public/` 和 Hexo 缓存 `db.json` 已被 Git 忽略，不需要提交。

确认无误后提交并推送源码：

```bash
git add .
git commit -m "Add new post"
git push origin main
```

`.github/workflows/pages.yml` 会自动安装依赖、执行构建，然后发布 `public/` 到 GitHub Pages。

首次发布前，需要在 GitHub 仓库的 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。

## 历史内容

- `source/2022/` 和 `source/2023/` 保留旧文章的静态 HTML，Hexo 会按原 URL 复制它们。
- `source/_data/legacy_posts.yml` 维护索引页底部的“历史归档”列表。
- `scripts/content-index.js` 生成 `/articles/` 索引页，Markdown 文章会按 TAG 自动分组。
