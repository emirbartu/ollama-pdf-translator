import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import ollama
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_translator.log"),
    ],
)
logger = logging.getLogger("pdf_translator")


class TranslationError(Exception):
    """Exception raised for translation-related errors."""
    pass


class PDFError(Exception):
    """Exception raised for PDF processing errors."""
    pass


@dataclass
class TranslationConfig:
    """Configuration for PDF translation.
    
    Attributes:
        source_lang: Source language code or name
        target_lang: Target language code or name
        model: Ollama model to use for translation
        fallback_font: Font to use if original font is not available
        batch_size: Number of text blocks to translate in one batch
        skip_pages: List of page numbers to skip (0-indexed)
        max_chunk_size: Maximum characters per translation request
    """
    source_lang: str
    target_lang: str
    model: str = "llama3.2"
    fallback_font: str = "helvetica"
    batch_size: int = 10
    skip_pages: List[int] = None
    max_chunk_size: int = 1000
    preserve_layout: bool = True
    
    def __post_init__(self):
        if self.skip_pages is None:
            self.skip_pages = []
        self.validate()

    def validate(self):
        """Validate configuration parameters."""
        if not isinstance(self.source_lang, str) or not isinstance(self.target_lang, str):
            raise ValueError("Language must be a string")
        if not self.source_lang.strip() or not self.target_lang.strip():
            raise ValueError("Language cannot be empty")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        if self.max_chunk_size < 100:
            raise ValueError("Max chunk size must be at least 100 characters")


class PDFTranslator:
    """PDF translator using Ollama LLM.
    
    This class handles the extraction, translation, and replacement of text in PDF files.
    """
    
    def __init__(self, config: TranslationConfig):
        """Initialize the PDF translator.
        
        Args:
            config: Translation configuration
        """
        self.config = config
        try:
            self._check_ollama_availability()
        except Exception as e:
            raise TranslationError(f"Ollama service unavailable: {e}")
    
    def _check_ollama_availability(self):
        """Check if Ollama service is available and model is loaded."""
        try:
            models = ollama.list()
            available_models = [model.get('name') for model in models.get('models', [])]
            if self.config.model not in available_models:
                logger.warning(f"Model {self.config.model} not found in available models: {available_models}")
                logger.info(f"Pulling model {self.config.model}...")
                ollama.pull(self.config.model)
        except Exception as e:
            raise TranslationError(f"Failed to connect to Ollama service: {e}")

    def translate_pdf(self, input_pdf: Union[str, Path], output_pdf: Union[str, Path]) -> None:
        """Translate a PDF file.
        
        Args:
            input_pdf: Path to input PDF file
            output_pdf: Path to output PDF file
        
        Raises:
            PDFError: If PDF processing fails
            TranslationError: If translation fails
        """
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)
        
        if not input_pdf.exists():
            raise PDFError(f"PDF file not found: {input_pdf}")
        
        # Create output directory if it doesn't exist
        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(input_pdf)
            total_pages = len(doc)
            logger.info(f"Processing PDF with {total_pages} pages")
            
            for page_num in tqdm(range(total_pages), desc="Translating pages"):
                if page_num in self.config.skip_pages:
                    logger.info(f"Skipping page {page_num+1}/{total_pages}")
                    continue
                
                logger.info(f"Processing page {page_num+1}/{total_pages}")
                page = doc[page_num]
                
                # Extract and translate text blocks
                blocks = self._extract_text_blocks(page)
                if not blocks:
                    logger.info(f"No text found on page {page_num+1}")
                    continue
                
                # Process blocks in batches
                for i in range(0, len(blocks), self.config.batch_size):
                    batch = blocks[i:i + self.config.batch_size]
                    for block in batch:
                        try:
                            # Skip blocks with very short text
                            if len(block['text'].strip()) < 3:
                                continue
                                
                            # Translate block text
                            translated_text = self._translate_text_with_chunking(block['text'])
                            
                            # Replace original text with translation
                            self._replace_block_text(page, block, translated_text)
                        except Exception as e:
                            logger.error(f"Error translating block: {str(e)}")
                            continue

            # Save the translated PDF
            doc.save(output_pdf)
            logger.info(f"Translated PDF saved as {output_pdf}")
        except Exception as e:
            raise PDFError(f"Error processing PDF: {e}")
        finally:
            if 'doc' in locals():
                doc.close()

    def _extract_text_blocks(self, page) -> List[Dict]:
        """Extract text blocks from a PDF page.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of text blocks with text, bounding boxes, font, and size
            
        Raises:
            PDFError: If text extraction fails
        """
        blocks = []
        try:
            # Get page text as dictionary structure
            page_dict = page.get_text("dict")
            
            # Process each text block
            for block in page_dict["blocks"]:
                if block["type"] == 0:  # Text block
                    # Collect text and properties from spans
                    full_text = ""
                    bboxes = []
                    fonts = set()
                    sizes = []
                    
                    for line in block["lines"]:
                        for span in line["spans"]:
                            full_text += span["text"] + " "
                            bboxes.append(span["bbox"])
                            fonts.add(span["font"])
                            sizes.append(span["size"])
                    
                    # Only add non-empty blocks
                    if full_text.strip():
                        # Find predominant font and average size
                        font = max(fonts, key=list(fonts).count) if fonts else self.config.fallback_font
                        size = sum(sizes) / len(sizes) if sizes else 11
                        
                        blocks.append({
                            "text": full_text.strip(),
                            "bboxes": bboxes,
                            "font": font,
                            "size": size
                        })
        except Exception as e:
            raise PDFError(f"Error extracting text: {e}")
            
        return blocks

    def _replace_block_text(self, page, block, translated_text: str) -> None:
        """Replace text in a PDF page.
        
        Args:
            page: PyMuPDF page object
            block: Text block with bounding boxes and properties
            translated_text: Translated text to insert
            
        Raises:
            PDFError: If text replacement fails
        """
        try:
            # Get font to use
            font_name = self._get_best_font(block['font'])
            
            # Use redaction annotations to replace text
            if self.config.preserve_layout:
                # When preserving layout, we use the original bounding boxes
                avg_bbox = self._calculate_average_bbox(block['bboxes'])
                rect = fitz.Rect(avg_bbox)
                
                # Create redaction annotation
                annot = page.add_redact_annot(
                    rect,
                    text=translated_text,
                    fontname=font_name,
                    fontsize=block['size'],
                    align=0  # Left alignment
                )
                
                # Apply redaction (this actually changes the PDF content)
                page.apply_redactions()
            else:
                # Alternative: insert text without strict layout preservation
                # First remove the original text
                for bbox in block['bboxes']:
                    rect = fitz.Rect(bbox)
                    page.add_redact_annot(rect)
                
                page.apply_redactions()
                
                # Then insert the translated text
                avg_bbox = self._calculate_average_bbox(block['bboxes'])
                rect = fitz.Rect(avg_bbox)
                
                # Insert the new text
                page.insert_textbox(
                    rect,
                    translated_text,
                    fontname=font_name,
                    fontsize=block['size'],
                    align=0  # Left alignment
                )
                
        except Exception as e:
            raise PDFError(f"Error replacing text: {e}")

    def _get_best_font(self, original_font: str) -> str:
        """Determine the best font to use.
        
        Args:
            original_font: Original font from PDF
            
        Returns:
            Font name to use
        """
        try:
            # Simple check if font exists
            fitz.Font(fontname=original_font)
            return original_font
        except:
            return self.config.fallback_font

    def _calculate_average_bbox(self, bboxes: List[List[float]]) -> List[float]:
        """Calculate the overall bounding box from multiple boxes.
        
        Args:
            bboxes: List of bounding boxes [x0, y0, x1, y1]
            
        Returns:
            Overall bounding box
        """
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return [x0, y0, x1, y1]

    def _translate_text_with_chunking(self, text: str) -> str:
        """Translate text with chunking for long texts.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        if not text.strip():
            return text
        
        # Split long text into chunks
        chunks = self._split_text(text, self.config.max_chunk_size)
        
        # Translate each chunk
        translated_chunks = []
        for chunk in chunks:
            try:
                translated = self._translate_text(chunk)
                translated_chunks.append(translated)
            except Exception as e:
                logger.error(f"Chunk translation error: {e}")
                # Fall back to original text for failed chunks
                translated_chunks.append(chunk)
        
        # Join translated chunks
        return " ".join(translated_chunks)

    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks of maximum length.
        
        Args:
            text: Text to split
            max_length: Maximum chunk length
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        # Try to split at sentence boundaries
        sentences = []
        current = ""
        
        # Split at common sentence endings
        for part in text.replace(". ", ".|").replace("! ", "!|").replace("? ", "?|").split("|"):
            if len(current + part) <= max_length:
                current += part + " "
            else:
                if current:
                    sentences.append(current.strip())
                
                # If a single sentence is too long, split it further
                if len(part) > max_length:
                    # Split at word boundaries
                    words = part.split()
                    current = ""
                    for word in words:
                        if len(current + word) <= max_length:
                            current += word + " "
                        else:
                            if current:
                                sentences.append(current.strip())
                            current = word + " "
                else:
                    current = part + " "
        
        if current:
            sentences.append(current.strip())
            
        return sentences

    def _translate_text(self, text: str) -> str:
        """Translate text using Ollama.
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails
        """
        if not text.strip():
            return text

        try:
            system_prompt = f"""
            You are a professional and precise translator. Translate the given text exactly from {self.config.source_lang} to {self.config.target_lang}. Follow these rules strictly:
            - Preserve all numbers, symbols, special characters, technical terms, file paths, and programming commands as they are.
            - Do not translate proper nouns, special names (e.g., trademarks, company names, product names), or URLs/links.
            - Maintain all formatting as closely as possible.
            - Contact information, such as email addresses, physical addresses, and phone numbers, must remain unchanged.
            - Do not introduce or remove any words, phrases, or punctuation not present in the original text.
            - If the source text includes untranslated words or phrases (e.g., placeholders like '[PLACEHOLDER]'), leave them untouched.
            
            Provide ONLY the translated text with no additional comments or explanations.
            """
            
            response = ollama.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this text: {text}"}
                ]
            )

            translated_text = response['message']['content'].strip()
            if not translated_text:
                raise TranslationError("Empty translation received")

            return translated_text

        except Exception as e:
            logger.error(f"Translation error: {e}")
            raise TranslationError(f"Failed to translate text: {e}")


def translate_pdf(
    input_pdf: Union[str, Path],
    output_pdf: Union[str, Path] = None,
    source_lang: str = "English",
    target_lang: str = "Spanish",
    model: str = "llama3.2",
    skip_pages: List[int] = None,
    preserve_layout: bool = True,
    fallback_font: str = "helvetica"
) -> Optional[Path]:
    """Translate a PDF file.
    
    Args:
        input_pdf: Path to input PDF file
        output_pdf: Path to output PDF file (default: input filename with _translated suffix)
        source_lang: Source language
        target_lang: Target language
        model: Ollama model to use
        skip_pages: List of page numbers to skip (0-indexed)
        preserve_layout: Whether to preserve original layout
        fallback_font: Font to use if original font is not available
        
    Returns:
        Path to translated PDF or None if translation failed
    """
    input_pdf = Path(input_pdf)
    
    if output_pdf is None:
        output_pdf = input_pdf.with_stem(f"{input_pdf.stem}_translated")
    else:
        output_pdf = Path(output_pdf)
    
    if output_pdf.exists():
        logger.warning(f"Warning: Overwriting existing file: {output_pdf}")
    
    try:
        config = TranslationConfig(
            source_lang=source_lang,
            target_lang=target_lang,
            model=model,
            skip_pages=skip_pages or [],
            preserve_layout=preserve_layout,
            fallback_font=fallback_font
        )
        
        translator = PDFTranslator(config)
        translator.translate_pdf(input_pdf, output_pdf)
        
        return output_pdf
        
    except (TranslationError, PDFError) as e:
        logger.error(f"Error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def main():
    """Command line interface for PDF translator."""
    parser = argparse.ArgumentParser(description="Translate PDF files using Ollama")
    parser.add_argument("input_pdf", help="Path to input PDF file")
    parser.add_argument("-o", "--output", help="Path to output PDF file")
    parser.add_argument("-s", "--source", default="English", help="Source language")
    parser.add_argument("-t", "--target", default="Spanish", help="Target language")
    parser.add_argument("-m", "--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--skip-pages", type=int, nargs="*", help="Pages to skip (0-indexed)")
    parser.add_argument("--no-preserve-layout", action="store_false", dest="preserve_layout", 
                        help="Do not preserve original layout")
    parser.add_argument("--font", default="helvetica", help="Fallback font")
    
    args = parser.parse_args()
    
    result = translate_pdf(
        input_pdf=args.input_pdf,
        output_pdf=args.output,
        source_lang=args.source,
        target_lang=args.target,
        model=args.model,
        skip_pages=args.skip_pages,
        preserve_layout=args.preserve_layout,
        fallback_font=args.font
    )
    
    if result is None:
        print("Translation failed. See log for details.")
        return 1
    else:
        print(f"Successfully created translated PDF: {result}")
        return 0


if __name__ == "__main__":
    main()