"""
Propositional Logic compute module.
Wraps PropositionalLogicEvaluator for GUI use.
"""

import re
from itertools import product


class PropositionalLogicEvaluator:
    """
    Evaluates propositional logic expressions.
    Supports: NOT, AND, OR, IMPLIES, IFF with Unicode and ASCII aliases.
    """

    def __init__(self):
        self.precedence = {
            '\u21d4': 1,  # ⇔
            '\u21d2': 2,  # ⇒
            '\u2228': 3,  # ∨
            '\u2227': 4,  # ∧
            '\u00ac': 5,  # ¬
        }

        self.operator_aliases = {
            '~': '\u00ac', 'NOT': '\u00ac', 'not': '\u00ac',
            '&': '\u2227', 'AND': '\u2227', 'and': '\u2227',
            '|': '\u2228', 'OR': '\u2228', 'or': '\u2228',
            '->': '\u21d2', '=>': '\u21d2', 'IMPLIES': '\u21d2', 'implies': '\u21d2',
            '<->': '\u21d4', '<=>': '\u21d4', 'IFF': '\u21d4', 'iff': '\u21d4',
        }

    def normalize_expression(self, expr):
        expr = expr.strip()
        for alias, standard in self.operator_aliases.items():
            expr = expr.replace(alias, standard)
        return expr

    def extract_propositions(self, expr):
        return set(re.findall(r'[A-Z]', expr))

    def tokenize(self, expr):
        tokens = []
        i = 0
        expr = expr.replace(' ', '')
        while i < len(expr):
            if expr[i] in '()':
                tokens.append(expr[i])
                i += 1
            elif expr[i] in self.precedence:
                tokens.append(expr[i])
                i += 1
            elif expr[i].isupper():
                tokens.append(expr[i])
                i += 1
            else:
                i += 1
        return tokens

    def infix_to_postfix(self, tokens):
        output = []
        stack = []
        for token in tokens:
            if token.isupper():
                output.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    output.append(stack.pop())
                if stack:
                    stack.pop()
            elif token in self.precedence:
                while (stack and stack[-1] != '(' and
                       stack[-1] in self.precedence and
                       self.precedence[stack[-1]] >= self.precedence[token]):
                    output.append(stack.pop())
                stack.append(token)
        while stack:
            output.append(stack.pop())
        return output

    def evaluate_postfix(self, postfix, model):
        stack = []
        for token in postfix:
            if token.isupper():
                stack.append(model[token])
            elif token == '\u00ac':
                stack.append(not stack.pop())
            elif token == '\u2227':
                r, l = stack.pop(), stack.pop()
                stack.append(l and r)
            elif token == '\u2228':
                r, l = stack.pop(), stack.pop()
                stack.append(l or r)
            elif token == '\u21d2':
                r, l = stack.pop(), stack.pop()
                stack.append(not l or r)
            elif token == '\u21d4':
                r, l = stack.pop(), stack.pop()
                stack.append(l == r)
        return stack[0] if stack else False

    def evaluate(self, expr, model):
        expr = self.normalize_expression(expr)
        tokens = self.tokenize(expr)
        postfix = self.infix_to_postfix(tokens)
        return self.evaluate_postfix(postfix, model)

    def generate_truth_table(self, expr):
        expr = self.normalize_expression(expr)
        propositions = sorted(self.extract_propositions(expr))
        if not propositions:
            return [], []
        results = []
        for values in product([False, True], repeat=len(propositions)):
            model = dict(zip(propositions, values))
            result = self.evaluate(expr, model)
            results.append(tuple(values) + (result,))
        return propositions, results

    def check_entailment_detailed(self, kb_exprs, query):
        kb_combined = ' \u2227 '.join(f'({expr})' for expr in kb_exprs)
        all_expr = self.normalize_expression(kb_combined + query)
        propositions = sorted(self.extract_propositions(all_expr))
        if not propositions:
            return {'entails': None, 'status': 'NOT SURE',
                    'explanation': 'No propositions found',
                    'kb_models': [], 'query_true_in_kb': [], 'counter_examples': []}
        kb_models, query_true_in_kb, counter_examples = [], [], []
        for values in product([False, True], repeat=len(propositions)):
            model = dict(zip(propositions, values))
            kb_true = all(self.evaluate(expr, model) for expr in kb_exprs)
            if kb_true:
                kb_models.append(model.copy())
                if self.evaluate(query, model):
                    query_true_in_kb.append(model.copy())
                else:
                    counter_examples.append(model.copy())
        if not kb_models:
            return {'entails': None, 'status': 'NOT SURE',
                    'explanation': 'KB is unsatisfiable (contradiction)',
                    'detail': 'KB is always false',
                    'kb_models': [], 'query_true_in_kb': [], 'counter_examples': []}
        if not counter_examples:
            return {'entails': True, 'status': 'TRUE',
                    'explanation': 'KB \u22a8 query',
                    'detail': f'Verified in all {len(kb_models)} model(s) where KB is true',
                    'kb_models': kb_models, 'query_true_in_kb': query_true_in_kb,
                    'counter_examples': []}
        return {'entails': False, 'status': 'FALSE',
                'explanation': 'KB \u22ad query',
                'detail': f'Found {len(counter_examples)} counter-example(s)',
                'kb_models': kb_models, 'query_true_in_kb': query_true_in_kb,
                'counter_examples': counter_examples}

    def check_multiple_queries(self, kb_exprs, queries):
        results = []
        for query in queries:
            result = self.check_entailment_detailed(kb_exprs, query)
            result['query'] = query
            results.append(result)
        return results

    def is_tautology(self, expr):
        _, results = self.generate_truth_table(expr)
        return all(row[-1] for row in results)

    def is_contradiction(self, expr):
        _, results = self.generate_truth_table(expr)
        return all(not row[-1] for row in results)

    def is_satisfiable(self, expr):
        _, results = self.generate_truth_table(expr)
        return any(row[-1] for row in results)

    def are_equivalent(self, expr1, expr2):
        props1 = self.extract_propositions(self.normalize_expression(expr1))
        props2 = self.extract_propositions(self.normalize_expression(expr2))
        all_props = sorted(props1.union(props2))
        if not all_props:
            return False
        for values in product([False, True], repeat=len(all_props)):
            model = dict(zip(all_props, values))
            if self.evaluate(expr1, model) != self.evaluate(expr2, model):
                return False
        return True


def _format_truth_table(propositions, results, expr):
    lines = []
    lines.append(f"Truth Table for: {expr}")
    lines.append("=" * (len(propositions) * 8 + 10))
    header = " | ".join(f"{p:^5}" for p in propositions) + f" | {'Result':^6}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in results:
        values = " | ".join(f"{'T' if v else 'F':^5}" for v in row[:-1])
        result = f"{'T' if row[-1] else 'F':^6}"
        lines.append(f"{values} | {result}")
    lines.append("=" * (len(propositions) * 8 + 10))
    return "\n".join(lines)


def _format_entailment(results, kb_exprs):
    lines = []
    lines.append("Knowledge Base:")
    for i, expr in enumerate(kb_exprs, 1):
        lines.append(f"  KB[{i}]: {expr}")
    lines.append("")

    for r in results:
        query = r.get('query', 'Query')
        status = r['status']
        sym = {
            'TRUE': '\u2713 TRUE (Entails)',
            'FALSE': '\u2717 FALSE (Does Not Entail)',
        }.get(status, '? NOT SURE')

        lines.append(f"Query: {query}")
        lines.append(f"Result: {sym}")
        lines.append(r.get('explanation', ''))
        lines.append(r.get('detail', ''))

        ce = r.get('counter_examples', [])
        if ce:
            shown = ce[:5]
            lines.append("Counter-example(s):")
            for i, model in enumerate(shown, 1):
                lines.append(f"  {i}. {dict(sorted(model.items()))}")
            if len(ce) > 5:
                lines.append(f"  ... and {len(ce) - 5} more")
        lines.append("")

    true_count = sum(1 for r in results if r['status'] == 'TRUE')
    false_count = sum(1 for r in results if r['status'] == 'FALSE')
    unsure_count = sum(1 for r in results if r['status'] == 'NOT SURE')
    lines.append("SUMMARY:")
    lines.append(f"  TRUE: {true_count}/{len(results)}")
    lines.append(f"  FALSE: {false_count}/{len(results)}")
    lines.append(f"  NOT SURE: {unsure_count}/{len(results)}")
    return "\n".join(lines)


def compute_logic(mode, expression="", expression2="", kb_lines=None, query_lines=None):
    """
    Compute propositional logic operations.

    mode: 'truth_table', 'entailment', 'check', 'equivalence'
    Returns standard result dict.
    """
    evaluator = PropositionalLogicEvaluator()

    if mode == 'truth_table':
        if not expression.strip():
            return {'text': 'ERROR: No expression provided.', 'summary_text': 'ERROR: No expression provided.'}
        props, results = evaluator.generate_truth_table(expression)
        if not props:
            return {'text': 'No propositions found.', 'summary_text': 'No propositions found.'}
        table_str = _format_truth_table(props, results, expression)
        # Analysis
        if evaluator.is_tautology(expression):
            table_str += "\n\nThis expression is a TAUTOLOGY (always true)"
        elif evaluator.is_contradiction(expression):
            table_str += "\n\nThis expression is a CONTRADICTION (always false)"
        else:
            table_str += "\n\nThis expression is CONTINGENT (sometimes true, sometimes false)"
        return {'text': table_str, 'summary_text': table_str}

    elif mode == 'entailment':
        kb = [line.strip() for line in (kb_lines or []) if line.strip()]
        queries = [line.strip() for line in (query_lines or []) if line.strip()]
        if not kb:
            return {'text': 'ERROR: No KB statements.', 'summary_text': 'ERROR: No KB statements.'}
        if not queries:
            return {'text': 'ERROR: No queries.', 'summary_text': 'ERROR: No queries.'}
        results = evaluator.check_multiple_queries(kb, queries)
        text = _format_entailment(results, kb)
        return {'text': text, 'summary_text': text}

    elif mode == 'check':
        if not expression.strip():
            return {'text': 'ERROR: No expression provided.', 'summary_text': 'ERROR: No expression provided.'}
        lines = []
        lines.append(f"Expression: {expression}")
        lines.append("")
        is_taut = evaluator.is_tautology(expression)
        is_contra = evaluator.is_contradiction(expression)
        is_sat = evaluator.is_satisfiable(expression)
        lines.append(f"Tautology:     {'YES' if is_taut else 'NO'}")
        lines.append(f"Contradiction: {'YES' if is_contra else 'NO'}")
        lines.append(f"Satisfiable:   {'YES' if is_sat else 'NO'}")
        text = "\n".join(lines)
        return {'text': text, 'summary_text': text}

    elif mode == 'equivalence':
        if not expression.strip() or not expression2.strip():
            return {'text': 'ERROR: Two expressions required.', 'summary_text': 'ERROR: Two expressions required.'}
        equiv = evaluator.are_equivalent(expression, expression2)
        lines = []
        lines.append(f"Expression 1: {expression}")
        lines.append(f"Expression 2: {expression2}")
        lines.append("")
        if equiv:
            lines.append(f"EQUIVALENT: {expression}  \u2261  {expression2}")
        else:
            lines.append(f"NOT EQUIVALENT: {expression}  \u2262  {expression2}")
        # Show both truth tables for comparison
        props1, res1 = evaluator.generate_truth_table(expression)
        props2, res2 = evaluator.generate_truth_table(expression2)
        lines.append("")
        lines.append(_format_truth_table(props1, res1, expression))
        lines.append("")
        lines.append(_format_truth_table(props2, res2, expression2))
        text = "\n".join(lines)
        return {'text': text, 'summary_text': text}

    return {'text': 'Unknown mode.', 'summary_text': 'Unknown mode.'}
