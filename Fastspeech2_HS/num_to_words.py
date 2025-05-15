from num2words import num2words

def num_to_word(text, language='hi'):
    """
    Convert numbers in text to words in the specified language.
    
    Args:
        text (str): Input text containing numbers
        language (str): Language code for conversion (default: 'hi' for Hindi)
        
    Returns:
        str: Text with numbers converted to words
    """
    try:
        # Split text into words
        words = text.split()
        result = []
        
        for word in words:
            # Try to convert if the word is a number
            try:
                num = float(word) if '.' in word else int(word)
                # Convert number to words in the specified language
                word_form = num2words(num, lang=language)
                result.append(word_form)
            except ValueError:
                # If not a number, keep the original word
                result.append(word)
        
        # Join the words back together
        return ' '.join(result)
    except Exception as e:
        print(f"Error in number to word conversion: {e}")
        return text  # Return original text if conversion fails 