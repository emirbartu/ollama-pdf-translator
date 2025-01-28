import fitz
import ollama
from dataclasses import dataclass
from typing import List, Optional, Tuple
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationError(Exception):
    pass

class PDFError(Exception):
    pass

@dataclass
class TranslationConfig:
    source_lang: str
    target_lang: str
    fallback_font: str = "notos" 

    def validate(self):
        if not isinstance(self.source_lang, str) or not isinstance(self.target_lang, str):
            raise ValueError("Language must be a string")
        if not self.source_lang.strip() or not self.target_lang.strip():
            raise ValueError("Language cannot be empty")

class PDFTranslator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        try:
            self.config.validate()
        except ValueError as e:
            raise TranslationError(f"Invalid configuration: {e}")

    def translate_pdf(self, input_pdf: str, output_pdf: str) -> None:
        if not os.path.exists(input_pdf):
            raise PDFError(f"PDF file not found: {input_pdf}")

        try:
            doc = fitz.open(input_pdf)
        except Exception as e:
            raise PDFError(f"Error opening PDF: {e}")

        for page in doc:
            blocks = self._extract_text_blocks(page)
            for block in blocks:
                try:
                    translated_text = self.translate_text(block['text'])
                    self._replace_block_text(page, block, translated_text)
                except Exception as e:
                    logger.error(f"Error translating block: {e}")
                    continue

        try:
            doc.save(output_pdf)
            logger.info(f"Translated PDF saved as {output_pdf}")
        except Exception as e:
            raise PDFError(f"Error saving translated PDF: {e}")
        finally:
            doc.close()

    def _extract_text_blocks(self, page) -> List[dict]:
        blocks = []
        try:
            page_blocks = page.get_text("dict")["blocks"]
            for block in page_blocks:
                if block["type"] == 0:
                    full_text = ""
                    bboxes = []
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text += span["text"] + " "
                            bboxes.append(span["bbox"])
                    if full_text.strip():
                        blocks.append({
                            "text": full_text.strip(),
                            "bboxes": bboxes,
                            "font": span["font"],
                            "size": span["size"]
                        })
        except Exception as e:
            raise PDFError(f"Error extracting text: {e}")
        return blocks

    def _replace_block_text(self, page, block, translated_text: str) -> None:
        try:
            font_name = self._get_best_font(block['font'])
            
            avg_bbox = self._calculate_average_bbox(block['bboxes'])
            
            rect = fitz.Rect(avg_bbox)
            annot = page.add_redact_annot(
                rect,
                text=translated_text,
                fontname=font_name,
                fontsize=block['size'],
                align=0
            )
            
            page.apply_redactions()
        except Exception as e:
            raise PDFError(f"Error replacing text: {e}")

    def _get_best_font(self, original_font: str) -> str:
        """Determine the best font to use"""
        try:
            # Simple check if font exists
            fitz.Font(fontname=original_font)
            return original_font
        except:
            return self.config.fallback_font

    def _calculate_average_bbox(self, bboxes: List[List[float]]) -> List[float]:
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return [x0, y0, x1, y1]

    def translate_text(self, text: str) -> str:
        if not text.strip():
            return text

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[
                    {
                        "role": "system",
                        "content": f"""
                        You are a professional and precise translator designed to handle PDF documents. Translate the given text exactly from {self.config.source_lang} to {self.config.target_lang}. Follow these rules strictly:
                        - Preserve all numbers, symbols, special characters, technical terms, file paths, and programming commands as they are. For example, do not translate 'git pull' into 'git ziehen'.
                        - Do not translate proper nouns, special names (e.g., trademarks, company names, product names), or URLs/links. For example, do not translate 'Google' or 'https://www.example.com'. 
                        - Maintain all formatting as closely as possible, including line breaks, lists, indentation, and any hierarchical structure of the text.
                        - Contact information, such as email addresses, physical addresses, and phone numbers, must remain unchanged. 
                        - Do not introduce or remove any words, phrases, or punctuation not present in the original text.
                        - If the source text includes untranslated words or phrases (e.g., placeholders like '[PLACEHOLDER]'), leave them untouched.
                        Ensure the translation is accurate, consistent, and adheres to these guidelines exactly without any explanations, comments, or additions in the output.
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Translate this text exactly as is: {text}"
                    }
                ]
            )

            translated_text = response['message']['content'].strip()
            if not translated_text:
                raise TranslationError("Empty translation received")

            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text


def translate_pdf(
    input_pdf: str,
    output_pdf: str,
    source_lang: str = "Italian",
    target_lang: str = "English"
) -> Optional[str]:
    try:
        if os.path.exists(output_pdf):
            print(f"Warning: Overwriting existing file: {output_pdf}")
            
        config = TranslationConfig(source_lang=source_lang, target_lang=target_lang)
        translator = PDFTranslator(config)
        
        translator.translate_pdf(input_pdf, output_pdf)
        
        return output_pdf
        
    except (TranslationError, PDFError) as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

if __name__ == "__main__":
    input_pdf = "input.pdf"
    output_pdf = "translated.pdf"
    
    result = translate_pdf(input_pdf, output_pdf)
    if result is None:
        print("Translation failed")
    else:
        print(f"Successfully created translated PDF: {result}")