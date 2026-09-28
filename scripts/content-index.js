'use strict';

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderPostList(posts) {
  if (!posts.length) return '<p>暂无文章。</p>';

  const items = posts.map(post => (
    `<li><time datetime="${escapeHtml(post.date)}">${escapeHtml(post.date)}</time>` +
    ` · <a href="${escapeHtml(post.path)}">${escapeHtml(post.title)}</a></li>`
  ));

  return `<ul>${items.join('')}</ul>`;
}

function joinUrlPath(root, path) {
  return `${root.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
}

hexo.extend.generator.register('content-index', locals => {
  const postsByTag = new Map();

  locals.posts.sort('-date').forEach(post => {
    const tags = post.tags.toArray().map(tag => tag.name);
    const tagNames = tags.length ? tags : ['其他'];
    const item = {
      title: post.title,
      date: post.date.format('YYYY-MM-DD'),
      path: joinUrlPath(hexo.config.root, post.path)
    };

    tagNames.forEach(tagName => {
      if (!postsByTag.has(tagName)) postsByTag.set(tagName, []);
      postsByTag.get(tagName).push(item);
    });
  });

  const tagSections = [...postsByTag.entries()]
    .sort(([left], [right]) => left.localeCompare(right, 'zh-CN'))
    .map(([tagName, posts]) => (
      '<section class="content-index-tag">' +
      `<h2>${escapeHtml(tagName)}</h2>${renderPostList(posts)}</section>`
    ))
    .join('');

  const legacyPosts = (locals.data.legacy_posts || [])
    .map(post => ({ title: post.title, date: post.date, path: post.path }))
    .sort((left, right) => right.date.localeCompare(left.date));

  const content = [
    '<p>当前及今后的 Markdown 文章按 TAG 自动整理在这里。</p>',
    tagSections,
    '<hr>',
    '<section id="legacy-archive">',
    '<h2>历史归档</h2>',
    '<p>以下是早期学习记录，保留当时发布的静态版本，不再继续更新。</p>',
    renderPostList(legacyPosts),
    '</section>'
  ].join('');

  return {
    path: 'articles/index.html',
    layout: 'page',
    data: {
      title: '索引',
      subtitle: '按 TAG 查找文章',
      banner_img: '/img/tree.png',
      content
    }
  };
});
