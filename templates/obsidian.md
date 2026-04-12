---
title: "{{ title }}"
author: "{{ author }}"
date: "{{ date }}"
category: "{{ category }}"
tags:
{% for tag in tags %}  - {{ tag }}
{% endfor %}
source: "微信视频号收藏"
---

# {{ title }}

> [!summary] 摘要
> {{ summary }}

## 📌 核心要点

{% for point in key_points %}
- {{ point }}
{% endfor %}

## 🔗 资源与工具

{% for resource in resources %}
- {{ resource }}
{% endfor %}

## ✅ 可行动建议

### 💻 开发应用
{% for item in action_items.dev %}
- [ ] {{ item }}
{% endfor %}

### 🏠 生活应用
{% for item in action_items.life %}
- [ ] {{ item }}
{% endfor %}

### 📝 技术总结
{% for item in action_items.tech_summary %}
- [ ] {{ item }}
{% endfor %}

---

## 📜 原始文稿

<details>
<summary>点击展开完整转录文字</summary>

{{ transcript }}

</details>
