package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"mote"

	"github.com/JohannesKaufmann/dom"
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"golang.org/x/net/html"
	"google.golang.org/genai"
)

const (
	hnURL        = "https://news.ycombinator.com"
	maxArticles  = 10
	fetchTimeout = 10 * time.Second
	llmTimeout   = 30 * time.Second
)

// Article represents a single HN story with its summary.
type Article struct {
	Title   string
	Link    string
	Content string // Markdown content
	Summary string
}

// callLLM simulates calling a language model to summarize content.
func callLLM(ctx context.Context, client *genai.Client, content string) (string, error) {
	if content == "" {
		return "[Summary: Empty content]", nil
	}

	result, err := client.Models.GenerateContent(ctx, "gemini-2.0-flash", genai.Text(
		fmt.Sprintf("Summarize the following content: %s", content)),
		nil)
	if err != nil {
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	return result.Text(), nil
}

// FetchHNFrontPageNode fetches the Hacker News front page HTML.
type FetchHNFrontPageNode struct {
	mote.BaseNode // Embed mote.BaseNode
}

func NewFetchHNFrontPageNode() *FetchHNFrontPageNode {
	return &FetchHNFrontPageNode{BaseNode: mote.NewBaseNode()}
}

// Act fetches the HTML content from news.ycombinator.com.
func (n *FetchHNFrontPageNode) Act(ctx context.Context, thought any) (any, error) {
	log.Println("Fetching HN front page...")
	client := &http.Client{Timeout: fetchTimeout}
	req, err := http.NewRequestWithContext(ctx, "GET", hnURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	// HN blocks default Go User-Agent, so use a common browser one
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch HN front page: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to fetch HN front page: status code %d", resp.StatusCode)
	}

	// Read the body content into a string to pass along
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read HN response body: %w", err)
	}
	log.Println("Fetched HN front page successfully.")
	return string(bodyBytes), nil
}

// Decide stores the fetched HTML in the shared state.
func (n *FetchHNFrontPageNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	htmlContent, ok := actResult.(string)
	if !ok {
		return "", fmt.Errorf("invalid actResult type: expected string, got %T", actResult)
	}
	state["hn_front_page_html"] = htmlContent
	return "default", nil
}

// ExtractLinksNode parses the HN HTML to find article links using standard library.
type ExtractLinksNode struct {
	mote.BaseNode // Embed mote.BaseNode
}

func NewExtractLinksNode() *ExtractLinksNode {
	return &ExtractLinksNode{BaseNode: mote.NewBaseNode()}
}

// Think retrieves the HTML from shared state.
func (n *ExtractLinksNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	htmlContent, ok := state["hn_front_page_html"].(string)
	if !ok || htmlContent == "" {
		return nil, fmt.Errorf("missing or invalid 'hn_front_page_html' in state")
	}
	return htmlContent, nil
}

func (n *ExtractLinksNode) Act(ctx context.Context, thought any) (any, error) {
	log.Println("Extracting article links using dom package...")
	htmlContent, ok := thought.(string)
	if !ok {
		return nil, fmt.Errorf("invalid thought type: expected string, got %T", thought)
	}

	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		return nil, fmt.Errorf("failed to parse HN HTML: %w", err)
	}

	var articles []Article

	// Find all table rows with class "athing" which represent articles
	athingRows := dom.FindAllNodes(doc, func(node *html.Node) bool {
		return dom.NodeName(node) == "tr" && dom.HasClass(node, "athing")
	})

	for _, row := range athingRows {
		if len(articles) >= maxArticles {
			break
		}

		// Find the <span class="titleline"> within the row
		titleLineSpan := dom.FindFirstNode(row, func(node *html.Node) bool {
			return dom.NodeName(node) == "span" && dom.HasClass(node, "titleline")
		})

		if titleLineSpan == nil {
			continue
		}

		// Find the <a> tag within the titleline span
		linkNode := dom.FindFirstNode(titleLineSpan, func(node *html.Node) bool {
			return dom.NodeName(node) == "a"
		})

		if linkNode == nil {
			continue
		}

		link := dom.GetAttributeOr(linkNode, "href", "")
		title := strings.TrimSpace(dom.CollectText(linkNode))

		if link != "" && title != "" {
			// Basic validation
			if !strings.HasPrefix(link, "http://") && !strings.HasPrefix(link, "https://") && !strings.HasPrefix(link, "item?id=") {
				log.Printf("Warning: Skipping item ('%s') with potentially relative or internal link: %s", title, link)
			} else {
				// Handle internal HN links
				if strings.HasPrefix(link, "item?id=") {
					link = hnURL + "/" + link
				}
				articles = append(articles, Article{Title: title, Link: link})
			}
		}
	}

	if len(articles) == 0 {
		log.Println("Warning: No articles extracted from HN front page using dom package.")
	} else {
		log.Printf("Extracted %d article links using dom package.", len(articles))
	}
	return articles, nil
}

// Decide stores the extracted links in shared state.
func (n *ExtractLinksNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	articles, ok := actResult.([]Article)
	if !ok {
		// Handle case where no articles were found (empty slice is okay)
		if actResult == nil {
			articles = []Article{} // Store empty slice
		} else {
			return "", fmt.Errorf("invalid actResult type: expected []Article, got %T", actResult)
		}
	}
	state["articles"] = articles
	return "default", nil
}

// ProcessArticlesNode fetches content for each link, cleans it, and calls LLM.
type ProcessArticlesNode struct {
	mote.BaseNode
	llmClient *genai.Client
}

func NewProcessArticlesNode(llmClient *genai.Client) *ProcessArticlesNode {
	return &ProcessArticlesNode{BaseNode: mote.NewBaseNode(), llmClient: llmClient}
}

// Think retrieves the list of articles from shared state.
func (n *ProcessArticlesNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	articles, ok := state["articles"].([]Article)
	if !ok {
		return nil, fmt.Errorf("missing or invalid 'articles' in state")
	}
	return articles, nil
}

// Act fetches, cleans, and summarizes each article.
func (n *ProcessArticlesNode) Act(ctx context.Context, thought any) (any, error) {
	log.Println("Processing articles...")
	articlesInput, ok := thought.([]Article)
	if !ok {
		return nil, fmt.Errorf("invalid thought type: expected []Article, got %T", thought)
	}

	var processedArticles []Article

	client := &http.Client{Timeout: fetchTimeout}

	for i, article := range articlesInput {
		// Check context cancellation before processing each article
		select {
		case <-ctx.Done():
			log.Printf("Article processing cancelled by context: %v", ctx.Err())
			return processedArticles, ctx.Err() // Return partially processed articles and context error
		default:
			// Continue
		}

		log.Printf("Processing article %d/%d: %s (%s)", i+1, len(articlesInput), article.Title, article.Link)

		// Special handling for HN's internal links (comments pages)
		if strings.Contains(article.Link, "item?id=") {
			log.Printf("Skipping summary for HN discussion page: %s", article.Link)
			article.Summary = "[Summary: HN Discussion Page]"
			processedArticles = append(processedArticles, article)
			continue
		}

		// Fetch article content
		req, err := http.NewRequestWithContext(ctx, "GET", article.Link, nil)
		if err != nil {
			log.Printf("Error creating request for %s: %v. Skipping.", article.Link, err)
			article.Summary = "[Summary: Error creating request]"
			processedArticles = append(processedArticles, article)
			continue
		}
		// Use a common user agent
		req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")

		resp, err := client.Do(req)
		if err != nil {
			log.Printf("Error fetching %s: %v. Skipping.", article.Link, err)
			article.Summary = "[Summary: Error fetching content]"
			processedArticles = append(processedArticles, article)
			continue
		}

		// Ensure body is closed even if we error out early
		// Use a function literal to capture the current 'resp'
		defer func(r *http.Response) {
			if r != nil && r.Body != nil {
				r.Body.Close()
			}
		}(resp)

		if resp.StatusCode != http.StatusOK {
			log.Printf("Error fetching %s: status code %d. Skipping.", article.Link, resp.StatusCode)
			article.Summary = fmt.Sprintf("[Summary: Error fetching - Status %d]", resp.StatusCode)
			processedArticles = append(processedArticles, article)
			continue
		}

		// Check content type - only process HTML
		contentType := resp.Header.Get("Content-Type")
		if !strings.Contains(contentType, "text/html") {
			log.Printf("Skipping non-HTML content (%s) for %s", contentType, article.Link)
			article.Summary = fmt.Sprintf("[Summary: Non-HTML content (%s)]", contentType)
			processedArticles = append(processedArticles, article)
			continue
		}

		// Read Body
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			log.Printf("Error reading body for %s: %v. Skipping.", article.Link, err)
			article.Summary = "[Summary: Error reading content]"
			processedArticles = append(processedArticles, article)
			resp.Body.Close() // Close body explicitly here
			continue
		}
		resp.Body.Close() // Close body after successful read

		htmlContent := string(bodyBytes)

		// Convert HTML to Markdown
		markdownContent, err := htmltomarkdown.ConvertString(htmlContent)
		if err != nil {
			log.Printf("Error converting HTML to Markdown for %s: %v. Skipping summary.", article.Link, err)
			article.Summary = "[Summary: Error converting content]"
			processedArticles = append(processedArticles, article)
			continue
		}
		article.Content = markdownContent // Store markdown content

		// Call LLM (stub) for summary
		llmCtx, cancel := context.WithTimeout(ctx, llmTimeout)
		summary, err := callLLM(llmCtx, n.llmClient, article.Content)
		cancel() // Release resources associated with llmCtx
		if err != nil {
			log.Printf("Error calling LLM for %s: %v.", article.Link, err)
			// Decide if we should stop the whole flow or just mark this article
			article.Summary = "[Summary: Error during summarization]"
			// Optionally: return nil, fmt.Errorf("LLM error on %s: %w", article.Link, err) // Stop flow
		} else {
			article.Summary = summary
		}

		processedArticles = append(processedArticles, article)
		time.Sleep(50 * time.Millisecond) // Small delay to avoid overwhelming sites/network
	}

	log.Printf("Processed %d articles.", len(processedArticles))
	return processedArticles, nil
}

// Decide stores the articles with summaries in shared state.
func (n *ProcessArticlesNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	articles, ok := actResult.([]Article)
	if !ok {
		// Handle case where processing might have yielded nil (e.g., context cancelled early)
		if actResult == nil {
			articles = []Article{} // Store empty slice
		} else {
			return "", fmt.Errorf("invalid actResult type: expected []Article, got %T", actResult)
		}
	}
	state["summarized_articles"] = articles
	return "default", nil
}

// DisplaySummariesNode prints the final summaries.
type DisplaySummariesNode struct {
	mote.BaseNode // Embed mote.BaseNode
}

func NewDisplaySummariesNode() *DisplaySummariesNode {
	return &DisplaySummariesNode{BaseNode: mote.NewBaseNode()}
}

// Think retrieves the summarized articles from shared state.
func (n *DisplaySummariesNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	articles, ok := state["summarized_articles"].([]Article)
	if !ok {
		// If processing failed or was cancelled, the key might not exist or be nil.
		// Check if 'articles' (from extraction) exists as a fallback for partial results.
		rawArticles, rawOk := state["articles"].([]Article)
		if rawOk {
			log.Println("Warning: 'summarized_articles' not found, displaying raw extracted articles.")
			return rawArticles, nil // Return raw articles if summaries aren't ready
		}
		return nil, fmt.Errorf("missing 'summarized_articles' or 'articles' in state")

	}
	return articles, nil
}

// Act prints the titles and summaries.
func (n *DisplaySummariesNode) Act(ctx context.Context, thought any) (any, error) {
	articles, ok := thought.([]Article)
	if !ok {
		return nil, fmt.Errorf("invalid thought type: expected []Article, got %T", thought)
	}

	if len(articles) == 0 {
		log.Println("No articles to display.")
		return nil, nil
	}

	fmt.Println("\n--- Hacker News Front Page Summaries ---") // Fixed string literal
	for i, article := range articles {
		fmt.Printf("\n%d. %s\n", i+1, article.Title)
		fmt.Printf("   Link: %s\n", article.Link)
		if article.Summary != "" {
			fmt.Printf("   Summary: %s\n", article.Summary)
		} else if article.Content == "" && article.Summary == "" {
			fmt.Println("   Summary: [Not processed or failed]")
		} else {
			fmt.Println("   Summary: [Summary not generated]") // Should ideally not happen with current logic
		}

	}
	fmt.Println("\n----------------------------------------") // Fixed string literal
	return nil, nil                                           // No further action needed
}

func main() {
	var apiKey = os.Getenv("GEMINI_API_KEY")
	var client, err = genai.NewClient(context.TODO(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}

	log.Println("Starting HackerNews Summarizer Flow...")

	// Create nodes
	fetchNode := NewFetchHNFrontPageNode()
	extractNode := NewExtractLinksNode()
	processNode := NewProcessArticlesNode(client)
	displayNode := NewDisplaySummariesNode()

	// Connect nodes: Fetch -> Extract -> Process -> Display
	fetchNode.AddSuccessor("default", extractNode).AddSuccessor("default", processNode).AddSuccessor("default", displayNode)

	// Create and run the flow
	flow := mote.NewFlow(fetchNode)    // Use mote.NewFlow
	initialState := mote.SharedState{} // Use mote.SharedState
	_, err = flow.Run(context.Background(), initialState)

	if err != nil {
		log.Fatalf("Flow execution failed: %v", err)
	} else {
		log.Println("Flow completed successfully.")
	}

}
