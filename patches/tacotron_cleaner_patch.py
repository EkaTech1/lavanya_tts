import sys
import os

def apply_patch():
    try:
        import tacotron_cleaner
        cleaner_init = os.path.join(os.path.dirname(tacotron_cleaner.__file__), '__init__.py')
        
        with open(cleaner_init, 'r') as f:
            content = f.read()
        
        # Fix the syntax warning by replacing 'is not' with '!='
        fixed_content = content.replace("s is not '_'", "s != '_'").replace("s is not '~'", "s != '~'")
        
        with open(cleaner_init, 'w') as f:
            f.write(fixed_content)
            
        return True
    except Exception as e:
        print(f"Failed to apply tacotron_cleaner patch: {str(e)}")
        return False 