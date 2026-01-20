from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import fitz
from PyPDF2 import PdfReader, PdfWriter
from io import BytesIO
from fpdf import FPDF
from PIL import Image
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
try:
    import img2pdf
    HAS_IMG2PDF = True
except ImportError:
    HAS_IMG2PDF = False

CONVERT_DPI = 300  # 平衡质量和文件大小（300 DPI 适合打印，更高的值可能导致内存/磁盘问题）

app = Flask(__name__)


# 图像去除水印函数（优化版）
def remove_watermark(image_path):
    """优化的水印去除函数，减少I/O操作"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: 无法读取 {image_path}")
            return False
        
        # 创建掩码（不需要高斯模糊可以更快）
        lower_bound = np.array([160, 160, 160])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        
        # 直接替换像素值
        img[mask == 255] = [255, 255, 255]
        
        # 写回文件（使用适当的压缩级别，避免文件过大导致写入失败）
        # 压缩级别 3 可以在质量和文件大小之间取得平衡
        success = cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        if not success:
            print(f"Error: 无法写入 {image_path}")
            return False
            
        return True
    except Exception as e:
        print(f"Error: 处理 {image_path} 时出错: {e}")
        return False


# 将PDF转换为图片，并保存到指定目录

def pdf_to_images(pdf_path, output_folder):
    """将PDF转换为图片并并行去除水印"""
    import shutil
    
    # 检查磁盘空间
    stat = shutil.disk_usage(output_folder)
    free_gb = stat.free / (1024**3)
    if free_gb < 1:
        print(f"警告: 磁盘空间不足 ({free_gb:.2f} GB 可用)，可能导致写入失败")
    
    images = []
    doc = fitz.open(pdf_path)
    dpi_scale = CONVERT_DPI / 72  # 使用全局DPI设置
    
    print(f"开始转换PDF，共 {doc.page_count} 页...")
    
    # 第一步：转换PDF为图片（带进度条）
    for page_num in tqdm(range(doc.page_count), desc="转换PDF页面", unit="页"):
        page = doc[page_num]
        # 使用设置的DPI进行转换
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale))
        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
        pix.save(image_path)
        images.append(image_path)
    
    doc.close()
    
    if not images:
        raise Exception("no images was generated.")
    
    # 第二步：并行去除水印（带进度条）
    print(f"开始去除水印（使用多线程加速）...")
    max_workers = min(os.cpu_count() or 4, len(images))  # 根据CPU核心数决定线程数
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_image = {executor.submit(remove_watermark, img_path): img_path 
                          for img_path in images}
        
        # 使用进度条显示完成情况
        with tqdm(total=len(images), desc="去除水印", unit="页") as pbar:
            for future in as_completed(future_to_image):
                image_path = future_to_image[future]
                try:
                    success = future.result()
                    if not success:
                        print(f"警告: {image_path} 处理失败")
                except Exception as exc:
                    print(f"错误: {image_path} 处理时出现异常: {exc}")
                pbar.update(1)
    
    print("水印去除完成！")
    return images


# 将图片合并为PDF

# 定义A4纸张在72dpi下的像素尺寸（宽度和高度）
A4_SIZE_PX_72DPI = (595, 842)


def images_to_pdf(image_paths, output_path):
    """将图片合并为PDF，优先使用img2pdf获得最佳质量"""
    
    # 方案1：使用img2pdf（推荐，质量最好，直接嵌入原图无损）
    if HAS_IMG2PDF:
        print("使用img2pdf生成高质量PDF...")
        try:
            with open(output_path, "wb") as f:
                # img2pdf直接将图像嵌入PDF，不进行重采样，保持原始质量
                f.write(img2pdf.convert(image_paths))
            print("PDF生成完成（使用img2pdf，质量最佳）")
            return
        except Exception as e:
            print(f"img2pdf失败: {e}，回退到FPDF方案")
    
    # 方案2：使用FPDF（备用方案）
    print("使用FPDF生成PDF...")
    pdf_writer = FPDF(unit='pt', format='A4')

    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size

            # 使用全局DPI设置
            dpi = CONVERT_DPI
            ratio = min(A4_SIZE_PX_72DPI[0] / width, A4_SIZE_PX_72DPI[1] / height)

            # 使用高质量重采样算法缩放图像以适应A4纸张
            img_resized = img.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)

            # 创建临时文件并写入图片数据（使用最高质量PNG）
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                # 使用无压缩PNG保持最高质量
                img_resized.save(temp_file.name, format='PNG', optimize=False, compress_level=0, quality=100)

            # 添加一页
            pdf_writer.add_page()

            # 使用临时文件路径添加图像到PDF
            pdf_writer.image(temp_file.name, x=0, y=0, w=A4_SIZE_PX_72DPI[0], h=A4_SIZE_PX_72DPI[1])

    # 清理临时文件
    for image_path in image_paths:
        _, temp_filename = os.path.split(image_path)
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    pdf_writer.output(output_path)
    print("PDF生成完成（使用FPDF）")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        pdf_path = 'uploads/uploaded_file.pdf'
        uploaded_file.save(pdf_path)
        return render_template('index.html', message='文件上传成功')


@app.route('/remove_watermark', methods=['GET'])
def remove_watermark_route():
    pdf_path = 'uploads/uploaded_file.pdf'
    output_folder = 'output_images'
    os.makedirs(output_folder, exist_ok=True)  # 创建输出目录（如果不存在）
    image_paths = pdf_to_images(pdf_path, output_folder)
    output_pdf_path = 'output_file.pdf'
    images_to_pdf(image_paths, output_pdf_path)
    return render_template('index.html', message='水印去除成功')


@app.route('/download')
def download():
    output_pdf_path = 'output_file.pdf'
    return send_file(output_pdf_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
