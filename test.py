from reasonflux import ReasonFlux

reasonflux = ReasonFlux(navigator_path='models/navigator',
                        template_matcher_path='jinaai/jina-embeddings-v3',
                        inference_path='models/inference_llm',
                        template_path='data/template_library.json')
problem = """Given a sequence {aₙ} satisfying a₁=3, and aₙ₊₁=2aₙ+5 (n≥1), find the general term formula aₙ"""
reasonflux.reason(problem)