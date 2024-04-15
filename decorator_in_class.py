import functools
def run_(txt):
    return "Running: " + txt

class RUN:
    
    def __init__(self, intermediate_text = "Intermediate"):
        self.intermediate_text = intermediate_text
    
    @staticmethod
    def run_wrapper(prefix = "", suffix = ""):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self = args[0]
                s = prefix
                s += self.intermediate_call()
                s += func(*args, **kwargs)
                s += suffix
                return s
            return wrapper
        return decorator
    
    def intermediate_call(self):
        return self.intermediate_text
    
    
    @run_wrapper(prefix = "Prefix<", suffix = ">Suffix")
    def run(self, txt = "Hello"):
        return txt
    
r = RUN()
print(r.run())