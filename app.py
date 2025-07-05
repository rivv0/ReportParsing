from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import re
import json
from difflib import get_close_matches
import PyPDF2
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Medical Lab Report Parser", version="1.0.0")

KNOWN_TESTS = [
    'FBS', 'PPBS', 'BLOOD UREA', 'S.CREATINE', 'CHOLESTEROL (TOTAL)', 'HDL CHOLESTEROL',
    'S.TRIGLYCERIDE', 'CALCIUM', 'URIC ACID', 'PHOSPHOROUS', 'S. BILIRUBIN', 'BICARBONATE',
    'GGTP', 'SGPT', 'SGOT', 'HBSAG', 'BONE MARROW', 'HIV', 'SP.AFB', 'CBC', 'HB.', 'URINE',
    'ESR', 'COOMB`S TEST', 'CPK - MB', 'LDH', 'CSF ROUTINE', 'URINE PREGNANCY TEST', 'G6PD',
    'BLOOD GROUP RH', 'PL. FLUID', 'RA WITH TITER', 'PTTK', 'RETIC COUNT', 'AEC', 'STOOL',
    'S.WIDAL', 'PLATELET COUNT', 'S AMYLASE', 'ASO TITERS', 'CRP', 'URINE PROTIEN 24 HOUR',
    'RBC INDICES', 'TCDC', 'PSMP', 'PSCM', 'PCV (Hematocrete)', 'S. PROTEIN', 'PSA', 'VDRL',
    'ALKPO4', 'S Sodium ([NA+)', 'MT TEST', 'BLOOD CULTURE', 'RBS', 'BT, CT.', 'CPK',
    'S.POTTASIUM (K+)', 'URINE CULTURE', 'TSH', 'SCIKLING TEST', 'S. Chloride (Cl-)',
    'PLASMA TESTOSTERONE', 'ASCTIES FLUID', 'GLYCOSYLATED HB (HbA1c)', 'SYPHILIS DIPSTICK',
    'LIPASE TEST', 'SERUM INSULIN', 'HB.ELECTROPHORESIS', 'S.LITHIUM', 'PROTHROMBIN TIME',
    'S.ACETONE', 'LE CELL TEST', 'ACID PHOSPHATASE', 'S.FERRITIN', 'TwentyFour Hr Urinary Calcium',
    'Lp (a)', 'C.E.A.', 'Homocysteine', 'HIV (ELISA)', 'F.N.A.C.', 'A.N.A.', 'Plasma Cortisol',
    'Pleural Fluid', 'Serum B12', 'Biopsy Histopathology', 'Urine Amino Acid',
    'S Valproic Acid level', 'ADA', 'Serum HCG titer', 'Free T3', 'Free T4', 'AntiHCV Antibody',
    'FSH', 'LH', 'Urine Microalbumin', 'Troponin T', 'S. Prolactin', 'Serum Protein Electrophoresis',
    'Urinary Potassium', 'HCV', 'Vitamin D3', 'T3', 'T4', 'Dengue IgG', 'Dengue IgM', 'Dengue NS1',
    'LDL Cholesterol', 'VLDL Cholesterol', 'S.Chloride', 'AFP', 'HCG', 'Average Sugar',
    'S. Globulin', 'Direct Bilirubin', 'Indirect Bilirubin', 'S. Widal O', 'S. Widal H',
    'S. Widal A', 'S. Widal B', 'TB IgM', 'TB IgG', 'T3,T4,TSH', 'CA 125', 'Chikungunya IgM',
    'Serum Vitamin D level', 'Dengue NS1&IgM&IgG', 'HBeAg', 'Alfa Feto Protein', 'S.Magnesium',
    'Ca. 19.9', 'Anti TPO Antibodies', 'UACR (Urinary Albumin Creat Ratio)'
]

class LabResultExtractor:
    def __init__(self):
        # Enhanced test mappings
        self.test_mappings = {
            # Hemoglobin variations
            'haemoglobin': 'HB.',
            'hemoglobin': 'HB.',
            'hb': 'HB.',
            
            # RBC and related tests
            'rdw': 'RBC INDICES',
            'red cell distribution width': 'RBC INDICES',
            'erythrocyte': 'RBC INDICES',
            'rbc': 'RBC INDICES',
            'mcv': 'RBC INDICES',
            'mch': 'RBC INDICES',
            'mchc': 'RBC INDICES',
            'pcv': 'PCV (Hematocrete)',
            'packed cell volume': 'PCV (Hematocrete)',
            'hematocrit': 'PCV (Hematocrete)',
            
            # WBC and CBC tests
            'lymphocytes': 'CBC',
            'lymphocyte': 'CBC',
            'neutrophils': 'CBC',
            'neutrophil': 'CBC',
            'monocytes': 'CBC',
            'monocyte': 'CBC',
            'eosinophils': 'CBC',
            'eosinophil': 'CBC',
            'basophils': 'CBC',
            'basophil': 'CBC',
            'leucocytes': 'CBC',
            'leucocyte': 'CBC',
            'leukocytes': 'CBC',
            'leukocyte': 'CBC',
            'wbc': 'CBC',
            'white blood cell': 'CBC',
            'total leucocytes': 'CBC',
            'absolute neutrophils': 'CBC',
            'absolute lymphocyte': 'CBC',
            'absolute monocyte': 'CBC',
            'absolute eosinophil': 'CBC',
            'absolute basophil': 'CBC',
            
            # Platelet tests
            'platelet': 'PLATELET COUNT',
            'platelets': 'PLATELET COUNT',
            'mpv': 'PLATELET COUNT',
            'mean platelet volume': 'PLATELET COUNT',
            
            # Liver function tests
            'sgpt': 'SGPT',
            'alt': 'SGPT',
            'sgot': 'SGOT',
            'ast': 'SGOT',
            
            # Kidney function tests
            'creatinine': 'S.CREATINE',
            'urea': 'BLOOD UREA',
            'blood urea': 'BLOOD UREA',
            
            # Electrolytes
            'potassium': 'S.POTTASIUM (K+)',
            'sodium': 'S Sodium ([NA+)',
            'chloride': 'S. Chloride (Cl-)',
            
            # Urine tests
            'urine': 'URINE',
            'colour': 'URINE',
            'color': 'URINE',
            'transparency': 'URINE',
            'appearance': 'URINE',
            'reaction': 'URINE',
            'specific gravity': 'URINE',
            'urine protein': 'URINE',
            'urine glucose': 'URINE',
            'urine sugar': 'URINE',
            'urine ketones': 'URINE',
            'bile pigments': 'URINE',
            'red blood cells': 'URINE',
            'pus cells': 'URINE',
            'epithelial cells': 'URINE',
            'crystals': 'URINE',
            'cast': 'URINE',
            
            # Other tests
            'malaria': 'SP.AFB',
            'malaria parasite': 'SP.AFB'
        }
    
    def _normalize_test_name(self, test_name: str) -> str:
        """Normalize test name for matching"""
        normalized = re.sub(r'[^\w\s]', '', test_name.lower())
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized
    
    def _find_matching_test(self, test_name: str) -> Optional[str]:
        """Find the best matching known test"""
        normalized = self._normalize_test_name(test_name)
        
        # Direct mapping check
        if normalized in self.test_mappings:
            return self.test_mappings[normalized]
        
        # Check for partial matches in mappings
        for key, value in self.test_mappings.items():
            if key in normalized or normalized in key:
                return value
        
        # Check against known tests directly
        for known_test in KNOWN_TESTS:
            known_normalized = self._normalize_test_name(known_test)
            if normalized == known_normalized:
                return known_test
            if normalized in known_normalized or known_normalized in normalized:
                return known_test
        
        # Fuzzy matching as last resort
        normalized_known = [self._normalize_test_name(test) for test in KNOWN_TESTS]
        matches = get_close_matches(normalized, normalized_known, n=1, cutoff=0.6)
        if matches:
            idx = normalized_known.index(matches[0])
            return KNOWN_TESTS[idx]
        
        return None
    
    def extract_lab_results(self, text: str) -> List[Dict[str, Any]]:
        """Extract lab results from text with high accuracy"""
        results = []
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)
        
        # Method 1: Extract from structured format (test name followed by value and unit)
        # This handles the format shown in your sample
        results.extend(self._extract_structured_format(text))
        
        # Method 2: Extract from tabular format
        results.extend(self._extract_tabular_format(text))
        
        # Method 3: Extract from line-by-line format
        results.extend(self._extract_line_format(text))
        
        # Remove duplicates and return
        return self._remove_duplicates(results)
    
    def _extract_structured_format(self, text: str) -> List[Dict[str, Any]]:
        """Extract from structured format like 'Test Name Value Unit Range'"""
        results = []
        
        # Pattern for: "Test Name Value Unit Reference"
        # Example: "RDW (Red Cell Distribution Width) 15.4 % 11.5-14.0"
        pattern = r'([A-Za-z\s\(\)\[\]\.,-]+?)\s+(\d+\.?\d*|\d+,\d+|Present\+?\d*|Absent|Not Detected|Clear|Pale Yellow|Trace)\s+([a-zA-Z/μ%³\.\-\+]+|cells/cu\.mm|mill/cu\.mm|10\^3/μL|gm/dL|U/L|mg/dL|mmol/L|fL|pg|/hpf)\s*(\d+\.?\d*-\d+\.?\d*|<=?\s*\d+\.?\d*|Absent|Clear|Pale Yellow)?'
        
        matches = re.findall(pattern, text)
        for match in matches:
            test_name = match[0].strip()
            value = match[1].strip()
            unit = match[2].strip()
            
            # Clean test name
            test_name = re.sub(r'\s+', ' ', test_name).strip()
            test_name = test_name.strip('.,()[]- ')
            
            # Skip if test name is too short
            if len(test_name) < 3:
                continue
            
            # Find matching test
            matched_test = self._find_matching_test(test_name)
            if matched_test:
                results.append({
                    'test_name': matched_test,
                    'value': value.replace(',', '') if ',' in value else value,
                    'unit': unit,
                    'raw_test_name': test_name
                })
        
        return results
    
    def _extract_tabular_format(self, text: str) -> List[Dict[str, Any]]:
        """Extract from tabular format"""
        results = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers and metadata
            if any(skip in line.lower() for skip in [
                'investigation', 'observed', 'value', 'unit', 'biological', 'reference', 'interval',
                'page', 'collected', 'registered', 'reported', 'dr.', 'md', 'pathology',
                'consultant', 'reg no', 'sample', 'processing', 'mrs.', 'age:', 'sex:', 'pid','remarks'
            ]):
                continue
            
            # Look for test results in the line
            # Pattern: test name followed by numerical value and optional unit
            parts = line.split()
            if len(parts) >= 2:
                # Try to find a numerical value in the parts
                for i, part in enumerate(parts):
                    if re.match(r'^\d+\.?\d*$|^\d+,\d+$|^Present\+?\d*$|^Absent$|^Not Detected$|^Clear$|^Pale Yellow$', part):
                        # Found a value, everything before this is likely the test name
                        test_name = ' '.join(parts[:i])
                        value = part
                        unit = parts[i+1] if i+1 < len(parts) and not re.match(r'^\d', parts[i+1]) else ""
                        
                        if len(test_name) >= 3:
                            matched_test = self._find_matching_test(test_name)
                            if matched_test:
                                results.append({
                                    'test_name': matched_test,
                                    'value': value.replace(',', '') if ',' in value else value,
                                    'unit': unit,
                                    'raw_test_name': test_name
                                })
                        break
        
        return results
    
    def _extract_line_format(self, text: str) -> List[Dict[str, Any]]:
        """Extract from simple line format"""
        results = []
        
        # Look for specific patterns in the text
        specific_patterns = [
            # Hemoglobin pattern
            (r'Haemoglobin\s*\(Hb\)\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'HB.'),
            (r'Hemoglobin\s*\(Hb\)\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'HB.'),
            (r'Hb\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'HB.'),
            
            # RBC Count
            (r'Erythrocyte\s*\(RBC\)\s*Count\s*(\d+\.?\d*)\s*([a-zA-Z/\.]+)', 'RBC INDICES'),
            (r'RBC\s*Count\s*(\d+\.?\d*)\s*([a-zA-Z/\.]+)', 'RBC INDICES'),
            
            # Platelet Count
            (r'Platelet\s*count\s*(\d+\.?\d*)\s*([a-zA-Z/μ³\^]+)', 'PLATELET COUNT'),
            
            # WBC Count
            (r'Total\s*Leucocytes\s*\(WBC\)\s*Count\s*(\d+\.?\d*|\d+,\d+)\s*([a-zA-Z/\.]+)', 'CBC'),
            (r'WBC\s*Count\s*(\d+\.?\d*|\d+,\d+)\s*([a-zA-Z/\.]+)', 'CBC'),
            
            # Liver function
            (r'SGPT\s*\(ALT\)\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'SGPT'),
            (r'ALT\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'SGPT'),
            
            # Kidney function
            (r'Creatinine,?\s*Serum\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'S.CREATINE'),
            (r'Creatinine\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'S.CREATINE'),
            
            # Electrolytes
            (r'Potassium,?\s*Serum\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'S.POTTASIUM (K+)'),
            (r'Potassium\s*(\d+\.?\d*)\s*([a-zA-Z/]+)', 'S.POTTASIUM (K+)'),
        ]
        
        for pattern, test_name in specific_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = match[0].replace(',', '') if ',' in match[0] else match[0]
                unit = match[1] if len(match) > 1 else ""
                
                results.append({
                    'test_name': test_name,
                    'value': value,
                    'unit': unit,
                    'raw_test_name': test_name
                })
        
        return results
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results"""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create a unique key based on test name and value
            key = (result['test_name'], result['value'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    """Extract text from PDF file"""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail="Unable to extract text from PDF")

@app.post("/extract-lab-results/")
async def extract_lab_results(file: UploadFile = File(...)):
    """Extract lab results from uploaded PDF file"""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        
        # Extract lab results
        extractor = LabResultExtractor()
        results = extractor.extract_lab_results(text)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Extracted {len(results)} lab results",
                "results": results,
                "debug_info": {
                    "total_characters": len(text),
                    "sample_text": text[:1000] + "..." if len(text) > 1000 else text
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/extract-lab-results-text/")
async def extract_lab_results_from_text(text_data: Dict[str, str]):
    """Extract lab results from text input"""
    
    if "text" not in text_data:
        raise HTTPException(status_code=400, detail="Text field is required")
    
    try:
        text = text_data["text"]
        
        # Extract lab results
        extractor = LabResultExtractor()
        results = extractor.extract_lab_results(text)
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Extracted {len(results)} lab results",
                "results": results
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing text: {str(e)}")

@app.get("/known-tests/")
async def get_known_tests():
    """Get list of all known test names"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "known_tests": KNOWN_TESTS,
            "total_tests": len(KNOWN_TESTS)
        }
    )

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Lab Report Parser"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)