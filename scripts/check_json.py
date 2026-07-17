#!/usr/bin/env python3
"""
JSON Diagnostic Tool - Analyzes JSON files line by line to find parsing errors
"""

import json
import sys

def check_json_file(filepath):
    """
    Reads a JSON file line by line and attempts to identify parsing errors.
    """
    print(f"Analyzing JSON file: {filepath}\n")
    
    # First, try to parse the entire file
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print("✓ File is valid JSON!")
        print(f"  Structure: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"  Keys: {len(data)} keys")
        elif isinstance(data, list):
            print(f"  Items: {len(data)} items")
        return
    except json.JSONDecodeError as e:
        print(f"✗ JSON parsing failed!")
        print(f"  Error: {e.msg}")
        print(f"  Line: {e.lineno}, Column: {e.colno}")
        print(f"  Position: {e.pos}\n")
    except Exception as e:
        print(f"✗ Error reading file: {e}\n")
        return
    
    # If parsing failed, analyze line by line
    print("\n" + "="*70)
    print("LINE-BY-LINE ANALYSIS")
    print("="*70 + "\n")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Track bracket/brace balance
    bracket_stack = []
    in_string = False
    escape_next = False
    
    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Check for common issues
        issues = []
        
        # Check for trailing commas before closing brackets
        if stripped.endswith(',}') or stripped.endswith(',]'):
            issues.append("Trailing comma before closing bracket")
        
        # Check for missing commas
        prev_line_num = line_num - 1
        while prev_line_num > 0:
            prev_stripped = lines[prev_line_num - 1].strip()
            if prev_stripped:
                break
            prev_line_num -= 1
        
        if prev_line_num > 0:
            prev_stripped = lines[prev_line_num - 1].strip()
            # Check if previous line should have comma
            if (prev_stripped and 
                not prev_stripped.endswith(',') and 
                not prev_stripped.endswith('{') and 
                not prev_stripped.endswith('[') and
                not prev_stripped.endswith(':') and
                stripped and
                not stripped.startswith('}') and
                not stripped.startswith(']')):
                if '"' in prev_stripped or prev_stripped.endswith('}') or prev_stripped.endswith(']'):
                    issues.append("Previous line might be missing comma")
        
        # Check for unescaped quotes
        if stripped.count('"') % 2 != 0 and '\\' not in stripped:
            issues.append("Odd number of quotes (possible unescaped quote)")
        
        # Check for single quotes instead of double quotes
        if "'" in stripped and '"' not in stripped:
            issues.append("Single quotes detected (JSON requires double quotes)")
        
        # Check for NaN, Infinity, undefined
        if any(word in stripped for word in ['NaN', 'Infinity', 'undefined', 'None', 'True', 'False']):
            issues.append("Invalid JSON literal (NaN/Infinity/undefined/None/True/False)")
        
        # Print line with issues
        if issues or line_num <= 10 or line_num >= len(lines) - 10:
            status = "⚠ " if issues else "  "
            print(f"{status}Line {line_num:4d}: {line.rstrip()}")
            for issue in issues:
                print(f"          └─ ⚠ {issue}")
    
    # Try to parse progressively to find exact break point
    print("\n" + "="*70)
    print("PROGRESSIVE PARSING")
    print("="*70 + "\n")
    
    full_text = ''.join(lines)
    for i in range(len(full_text), 0, -max(1, len(full_text)//100)):
        try:
            json.loads(full_text[:i])
            print(f"✓ Valid JSON up to character {i} of {len(full_text)}")
            # Find corresponding line
            char_count = 0
            for line_num, line in enumerate(lines, 1):
                char_count += len(line)
                if char_count >= i:
                    print(f"  (approximately line {line_num})")
                    break
            
            # Show context around failure
            next_chars = full_text[i:i+100]
            print(f"\n  Next characters after valid JSON:")
            print(f"  {repr(next_chars)}")
            break
        except json.JSONDecodeError:
            continue
    
    # Try to provide specific fix suggestions
    print("\n" + "="*70)
    print("SUGGESTIONS")
    print("="*70 + "\n")
    
    suggestions = []
    
    # Check for common issues
    if full_text.count('{') != full_text.count('}'):
        diff = full_text.count('{') - full_text.count('}')
        if diff > 0:
            suggestions.append(f"• Add {diff} closing brace(s) '}}'")
        else:
            suggestions.append(f"• Remove {-diff} extra closing brace(s) '}}'")
    
    if full_text.count('[') != full_text.count(']'):
        diff = full_text.count('[') - full_text.count(']')
        if diff > 0:
            suggestions.append(f"• Add {diff} closing bracket(s) ']'")
        else:
            suggestions.append(f"• Remove {-diff} extra closing bracket(s) ']'")
    
    if ',}' in full_text or ',]' in full_text:
        suggestions.append("• Remove trailing commas before closing brackets")
    
    if suggestions:
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("Run the file through a JSON validator for more details.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to the found JSON file
        filepath = "papers/SpatioTemporalTransformation/analysis/data/stp/data_stp.json"
    
    check_json_file(filepath)
