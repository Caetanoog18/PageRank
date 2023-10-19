import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Dictionary of probability distribution
    prop_distribution = {}
    num_pages = len(corpus)
    num_links = len(corpus[page])

    if num_links > 0:

        # With probability damping_factor, the random surfer should randomly choose one of the links from page with equal probability.
        # With probability 1 - damping_factor, the random surfer should randomly choose one of all pages in the corpus with equal probability.
        for i in corpus:
            prop_distribution[i] = (1 - damping_factor) / num_pages

        for i in corpus[page]:
            prop_distribution[i] += damping_factor / num_links

    # If page has no outgoing links, then transition_model should return a probability distribution that chooses randomly among all pages with equal probability.
    else:
        for i in corpus:
            prop_distribution[i] = 1/num_pages

    return prop_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
 
    page_rank = {key: 0 for key in corpus.keys()}

    # The first sample needs to be a random choice
    next_page = random.choice(list(corpus.keys()))

    for _ in range(1, n):
        page_rank[next_page] +=1

        model = transition_model(corpus, next_page, damping_factor)

        next_page = random.choices(list(model), model.values(), k=1)[0]


    for key in page_rank.keys():
        page_rank[key] /= n

    return page_rank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    calculated_pr = {}
    condition = True

    # Initialize the PageRank dictionary for each page with an equal initial probability for all pages
    page_rank = {page: 1 / N for page in corpus}

    while condition:
        for current_page in page_rank:
            sum_links = 0

            for page in corpus:
                if current_page != page and current_page in corpus[page]:
                    sum_links += page_rank[page] / len(corpus[page])

                if not corpus[page]:
                    sum_links += page_rank[page] / N

            new_pr = ((1-damping_factor) / N) + damping_factor * sum_links
            calculated_pr[current_page] = new_pr

        condition = False

        for current_page in page_rank:
            if abs(page_rank[current_page] - calculated_pr[current_page]) > 0.001:
                condition = True

            page_rank[current_page] = calculated_pr[current_page]

    return page_rank

if __name__ == "__main__":
    main()