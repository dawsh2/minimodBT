import ast

def extract_functions_and_docstrings(filepath):
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)
    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name
            args = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node)
            functions.append((func_name, args, docstring))

    return functions

# Example usage:
file_path = "main.py"  # or any other Python file
for name, args, doc in extract_functions_and_docstrings(file_path):
    print(f"Function: {name}({', '.join(args)})")
    print(f"Docstring: {doc}\n")
