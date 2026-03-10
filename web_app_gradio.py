# -*- coding: utf-8 -*-
# !/usr/bin/python
# @Time    : 2025/12/04
# @Function: 中文文本智能纠错系统 - Gradio Web 界面

import gradio as gr
import os
import sys
import pandas as pd
from datetime import datetime

# 设置标准输出编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 兼容
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ============== 环境配置 ==============
# 禁用所有代理设置
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['ALL_PROXY'] = ''
os.environ['all_proxy'] = ''

# 启用文本纠错模块
os.environ["MACRO_CORRECT_FLAG_CSC_TOKEN"] = "1"

# 导入纠错模块
try:
    from macro_correct import correct
    print("[成功] 成功导入 macro_correct.correct 函数")
except ImportError as e:
    print(f"[失败] 导入失败: {e}")
    print("请确保已安装 macro-correct 包：pip install -e .")
    sys.exit(1)

# ============== 核心功能函数 ==============

def format_errors(errors):
    """格式化错误信息为表格"""
    if not errors:
        return None
    
    # 转换为 DataFrame
    df = pd.DataFrame(errors, columns=['原字符', '纠正字符', '位置', '置信度'])
    df['置信度'] = df['置信度'].apply(lambda x: f"{x:.2%}")
    return df


def highlight_differences(source, target, errors):
    """高亮显示文本差异"""
    if not errors:
        return source, target
    
    # 为原文和纠正文本添加高亮标记
    source_html = []
    target_html = []
    error_positions = {err[2]: (err[0], err[1]) for err in errors}
    
    for i, char in enumerate(source):
        if i in error_positions:
            source_html.append(f'<span style="background-color: #ffcccc; font-weight: bold;">{char}</span>')
        else:
            source_html.append(char)
    
    for i, char in enumerate(target):
        if i in error_positions:
            target_html.append(f'<span style="background-color: #ccffcc; font-weight: bold;">{char}</span>')
        else:
            target_html.append(char)
    
    return ''.join(source_html), ''.join(target_html)


def correct_single_text(text, threshold, flag_confusion):
    """单文本纠错"""
    if not text or not text.strip():
        return "请输入需要纠错的文本", "", None, ""
    
    try:
        # 执行纠错
        result = correct(
            [text],
            threshold=threshold,
            flag_confusion=flag_confusion,
            flag_prob=True
        )[0]
        
        source = result['source']
        target = result['target']
        errors = result['errors']
        
        # 格式化结果
        if errors:
            # 高亮差异
            source_html, target_html = highlight_differences(source, target, errors)
            
            # 错误统计
            stats = f"📊 **检测到 {len(errors)} 处错误**\n\n"
            stats += f"- 原文长度: {len(source)} 字符\n"
            stats += f"- 错误率: {len(errors)/len(source)*100:.2f}%"
            
            # 格式化错误表格
            error_df = format_errors(errors)
            
            return source_html, target_html, error_df, stats
        else:
            return source, target, None, "✅ **未检测到错误，文本正确！**"
            
    except Exception as e:
        return f"❌ 错误: {str(e)}", "", None, ""


def correct_batch_text(text, threshold, flag_confusion):
    """批量文本纠错（按行分割）"""
    if not text or not text.strip():
        return "请输入需要纠错的文本（每行一个）", None
    
    try:
        # 按行分割
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # 批量纠错
        results = correct(
            lines,
            threshold=threshold,
            flag_confusion=flag_confusion,
            flag_prob=True
        )
        
        # 格式化结果
        output_lines = []
        all_errors = []
        
        for i, result in enumerate(results):
            source = result['source']
            target = result['target']
            errors = result['errors']
            
            output_lines.append(f"**文本 {i+1}:**")
            output_lines.append(f"- 原文: {source}")
            output_lines.append(f"- 纠正: {target}")
            
            if errors:
                output_lines.append(f"- 错误数: {len(errors)}")
                all_errors.extend([(i+1, err[0], err[1], err[2], f"{err[3]:.2%}") for err in errors])
            else:
                output_lines.append("- ✅ 无错误")
            output_lines.append("")
        
        # 生成错误汇总表
        error_summary = None
        if all_errors:
            error_summary = pd.DataFrame(
                all_errors,
                columns=['文本序号', '原字符', '纠正字符', '位置', '置信度']
            )
        
        return "\n".join(output_lines), error_summary
        
    except Exception as e:
        return f"❌ 错误: {str(e)}", None


def process_file(file, threshold, flag_confusion):
    """处理上传的文件"""
    if file is None:
        return "请上传文件", None
    
    try:
        # 读取文件内容
        with open(file.name, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 按行处理
        return correct_batch_text(content, threshold, flag_confusion)
        
    except Exception as e:
        return f"❌ 文件处理错误: {str(e)}", None


# ============== Gradio 界面 ==============

# 自定义 CSS
custom_css = """
.container {
    max-width: 1200px;
    margin: auto;
}
.highlight-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.error-text {
    background-color: #ffcccc;
    padding: 2px 4px;
    border-radius: 3px;
}
.correct-text {
    background-color: #ccffcc;
    padding: 2px 4px;
    border-radius: 3px;
}
"""

# 创建 Gradio 界面
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="green",
    ),
    css=custom_css,
    title="中文文本智能纠错系统"
) as demo:
    
    # 标题和说明
    gr.Markdown("""
    # 🔍 中文文本智能纠错系统
    ### 基于深度学习的中文拼写纠错工具，支持各领域文本智能纠错
    """)
    
    # 参数配置区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 参数设置")
            threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.55,
                step=0.05,
                label="置信度阈值",
                info="低于此阈值的纠正会被过滤"
            )
            flag_confusion = gr.Checkbox(
                value=True,
                label="启用混淆词典",
                info="使用预定义的混淆词典提高准确率"
            )
    
    # Tab 选项卡
    with gr.Tabs():
        
        # ============== Tab 1: 单文本纠错 ==============
        with gr.Tab("📝 单文本纠错"):
            gr.Markdown("### 输入单条文本进行纠错")
            
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        lines=5,
                        placeholder="请输入需要纠错的文本...\n例如：机七学习是人工智能领遇最能体现智能的一个分知",
                        label="原始文本",
                        show_label=True
                    )
                    
                    with gr.Row():
                        correct_btn = gr.Button("🔍 开始纠错", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", size="lg")
            
            with gr.Row():
                with gr.Column():
                    output_source = gr.HTML(label="原文（标记错误）")
                    output_target = gr.HTML(label="纠正后文本")
            
            with gr.Row():
                with gr.Column(scale=2):
                    error_table = gr.Dataframe(
                        label="🔎 错误详情",
                        headers=['原字符', '纠正字符', '位置', '置信度'],
                        datatype=['str', 'str', 'number', 'str'],
                        row_count=5
                    )
                with gr.Column(scale=1):
                    stats_output = gr.Markdown(label="📊 统计信息")
            
            # 示例
            gr.Markdown("### 💡 示例文本（点击使用）")
            gr.Examples(
                examples=[
                    ["真麻烦你了。希望你们好好的跳无"],
                    ["少先队员因该为老人让坐"],
                    ["机七学习是人工智能领遇最能体现智能的一个分知"],
                    ["一只小鱼船浮在平净的河面上"],
                    ["他法语说的很好，的语也不错"],
                    ["遇到一位很棒的奴生跟我疗天"],
                    ["我们为这个目标努力不解"],
                    ["这一条次,我选择了一条与往常不同的路线"],
                ],
                inputs=input_text,
                label="点击示例自动填充"
            )
            
            # 绑定事件
            correct_btn.click(
                fn=correct_single_text,
                inputs=[input_text, threshold, flag_confusion],
                outputs=[output_source, output_target, error_table, stats_output]
            )
            
            clear_btn.click(
                fn=lambda: ("", "", None, ""),
                outputs=[input_text, output_target, error_table, stats_output]
            )
        
        # ============== Tab 2: 批量纠错 ==============
        with gr.Tab("📚 批量纠错"):
            gr.Markdown("### 输入多行文本进行批量纠错（每行一个文本）")
            
            batch_input = gr.Textbox(
                lines=10,
                placeholder="请输入多行文本，每行一个需要纠错的句子...\n例如：\n真麻烦你了。希望你们好好的跳无\n少先队员因该为老人让坐\n机七学习是人工智能领遇最能体现智能的一个分知",
                label="批量输入文本"
            )
            
            batch_btn = gr.Button("🔍 批量纠错", variant="primary", size="lg")
            
            batch_output = gr.Markdown(label="纠错结果")
            batch_error_table = gr.Dataframe(
                label="🔎 错误汇总",
                headers=['文本序号', '原字符', '纠正字符', '位置', '置信度']
            )
            
            batch_btn.click(
                fn=correct_batch_text,
                inputs=[batch_input, threshold, flag_confusion],
                outputs=[batch_output, batch_error_table]
            )
        
        # ============== Tab 3: 文件上传 ==============
        with gr.Tab("📄 文件上传"):
            gr.Markdown("### 上传文本文件进行批量纠错")
            
            file_input = gr.File(
                label="上传文本文件（.txt）",
                file_types=[".txt"],
                type="filepath"
            )
            
            file_btn = gr.Button("🔍 处理文件", variant="primary", size="lg")
            
            file_output = gr.Markdown(label="处理结果")
            file_error_table = gr.Dataframe(
                label="🔎 错误汇总",
                headers=['文本序号', '原字符', '纠正字符', '位置', '置信度']
            )
            
            file_btn.click(
                fn=process_file,
                inputs=[file_input, threshold, flag_confusion],
                outputs=[file_output, file_error_table]
            )
        
    
    # 页脚
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666;">
        <p>中文文本智能纠错系统 | Powered by Deep Learning</p>
    </div>
    """)


# ============== 启动应用 ==============
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 正在启动中文文本智能纠错系统...")
    print("=" * 60)
    print("\n📌 系统信息:")
    print("  - 模型: 深度学习纠错模型")
    print("  - 端口: 7860")
    print("  - 访问: http://localhost:7860")
    print("\n⏳ 首次启动需要加载模型，请稍候...\n")
    
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 设为 True 可生成公网链接
        show_error=True,         # 显示详细错误
        quiet=False,             # 显示启动信息
        inbrowser=True,          # 自动打开浏览器
    )

