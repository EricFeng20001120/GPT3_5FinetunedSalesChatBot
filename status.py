total_tokens_used = 0
total_cost = 0

memory_size = 3

multivector_retriever_use_summary = False

cache_dir = "./huggingface_cache"

resource_dir = "resources/"
product_path = resource_dir + "product.csv"

supported_file_types = [
    '.txt',  '.md',   '.py',   '.c',    '.cpp',
    '.h',    '.java', '.js',   '.html', '.css',
    '.xml',  '.json', '.yml',  '.yaml', '.php',
    '.m',    '.v',    '.vhdl', '.sv',   '.sh',
    '.bat',  '.pl',   '.sql',  '.lua',  '.log', '.pdf'] 

# pricing
gpt3_5_in  = 0.0010 / 1000
gpt3_5_gen = 0.0020 / 1000
gpt4_in  = 0.03 / 1000
gpt4_gen = 0.06 / 1000