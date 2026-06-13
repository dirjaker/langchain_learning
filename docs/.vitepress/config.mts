import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'LangChain Learning',
  description: 'LangChain + LangGraph 学习实战',
  base: '/langchain_learning/',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '速查表', link: '/CHEATSHEET' },
      { text: '技术文档', link: '/TECHNICAL_DOC' },
      { text: 'GitHub', link: 'https://github.com/dirjaker/langchain_learning' }
    ],
    sidebar: [
      {
        text: '开始',
        items: [
          { text: '首页', link: '/' },
        ]
      },
      {
        text: '文档',
        items: [
          { text: '面试速查表', link: '/CHEATSHEET' },
          { text: '技术文档', link: '/TECHNICAL_DOC' },
          { text: '技术设计文档', link: '/技术文档' },
        ]
      },
      {
        text: '代码示例',
        items: [
          { text: 'LangChain 版本演进', link: '/01_langchain_evolution' },
          { text: 'LangGraph 基础', link: '/02_langgraph_basics' },
          { text: '实战示例', link: '/03_practical_examples' },
          { text: '核心组件', link: '/04_core_components' },
          { text: 'LangGraph 高级', link: '/05_langgraph_advanced' },
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/dirjaker/langchain_learning' }
    ],
    search: {
      provider: 'local'
    },
    outline: {
      label: '页面导航'
    },
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
    lastUpdated: {
      text: '最后更新于'
    }
  },
  lastUpdated: true,
  markdown: {
    lineNumbers: true
  }
})
