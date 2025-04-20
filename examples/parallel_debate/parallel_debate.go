package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"mote"

	"google.golang.org/genai" // Using genai for consistency, even if stubbed
)

// DebateArgument holds the refined points for a specific stance.
type DebateArgument struct {
	Stance string
	Points []string
}

// Constants for main flow shared state keys
const (
	StateTopic          = "topic"
	StateStances        = "stances"
	StateFinalArgsMap   = "final_arguments_map" // map[string]DebateArgument (shared between main and sub-flows)
	StateMutex          = "mutex"               // *sync.Mutex (shared)
	StateWaitGroup      = "waitGroup"           // *sync.WaitGroup (for main flow)
	StateFinalJudgement = "finalJudgement"
	StateSubFlowsMap    = "sub_flows_map" // map[string]*mote.Flow (Added)
)

// Constants for sub-flow shared state keys
const (
	SubStateStance        = "stance"
	SubStateTopic         = "topic"
	SubStateCurrentArgs   = "current_arguments"
	SubStateIteration     = "iteration"
	SubStateMaxIterations = "max_iterations"
	SubStateSharedArgsMap = "shared_arguments_map"
	SubStateSharedMutex   = "shared_mutex"
)

// --- Mock LLM Call ---
func callLLM(ctx context.Context, client *genai.Client, prompt string, args ...any) string {
	fullPrompt := fmt.Sprintf(prompt, args...)

	// Use a timeout for the LLM call itself to prevent hangs
	llmCtx, cancel := context.WithTimeout(ctx, 15*time.Second) // 15 sec timeout for LLM
	defer cancel()

	result, err := client.Models.GenerateContent(llmCtx, "gemini-2.0-flash", genai.Text(fullPrompt), nil)
	if err != nil {
		if llmCtx.Err() == context.DeadlineExceeded {
			log.Printf("LLM call timed out: %v", err)
			return "LLM_TIMEOUT_ERROR"
		}
		log.Printf("Error calling LLM: %v", err)
		return "LLM_GENERATION_ERROR"
	}

	// Return the raw text, trimmed of whitespace
	responseText := strings.TrimSpace(result.Text())
	if responseText == "" {
		return "LLM_EMPTY_RESPONSE_ERROR"
	}

	return responseText
}

// --- Sub-Flow Nodes (Argument Refinement Loop) ---

// GenerateInitialArgumentNode: Starts the sub-flow for a stance.
type GenerateInitialArgumentNode struct {
	mote.BaseNode
	llmClient *genai.Client
}

func NewGenerateInitialArgumentNode(client *genai.Client) *GenerateInitialArgumentNode {
	return &GenerateInitialArgumentNode{BaseNode: mote.NewBaseNode(), llmClient: client}
}

func (n *GenerateInitialArgumentNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	topic, okTopic := state[SubStateTopic].(string)
	stance, okStance := state[SubStateStance].(string)
	if !okTopic || !okStance {
		return nil, fmt.Errorf("missing topic or stance in sub-flow state")
	}
	return []string{topic, stance}, nil
}

func (n *GenerateInitialArgumentNode) Act(ctx context.Context, thought any) (any, error) {
	params, _ := thought.([]string)
	topic, stance := params[0], params[1]
	log.Printf("[%s] Generating initial arguments...", stance)
	argsResult := callLLM(ctx, n.llmClient, "Generate initial arguments for stance '%s' on topic '%s'", stance, topic)
	return argsResult, nil
}

func (n *GenerateInitialArgumentNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	argsResult, ok := actResult.(string)
	if !ok {
		return "", fmt.Errorf("invalid actResult type for initial args: %T", actResult)
	}

	// Check for errors returned by callLLM
	if strings.HasPrefix(argsResult, "LLM_") {
		return "", fmt.Errorf("LLM error generating initial args: %s", argsResult)
	}

	// Split the raw string into points for storage
	argsPoints := strings.Split(argsResult, "\n")
	var cleanedPoints []string
	for _, point := range argsPoints {
		cleaned := strings.TrimSpace(point)
		if cleaned != "" {
			cleanedPoints = append(cleanedPoints, cleaned)
		}
	}
	if len(cleanedPoints) == 0 { // Handle case where splitting yields nothing
		cleanedPoints = []string{argsResult} // Store the original string as one point
	}

	state[SubStateCurrentArgs] = cleanedPoints // Store as []string
	state[SubStateIteration] = 1
	log.Printf("[%s] Initial arguments generated. Iteration: %d", state[SubStateStance], state[SubStateIteration])
	return "evaluate", nil // Transition back to Evaluate node
}

// EvaluateArgumentNode: Decides if arguments are sufficient or need refinement.
type EvaluateArgumentNode struct {
	mote.BaseNode
	llmClient *genai.Client
}

func NewEvaluateArgumentNode(client *genai.Client) *EvaluateArgumentNode {
	return &EvaluateArgumentNode{BaseNode: mote.NewBaseNode(), llmClient: client}
}

func (n *EvaluateArgumentNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	args, ok := state[SubStateCurrentArgs].([]string)
	if !ok {
		return nil, fmt.Errorf("missing current_arguments in sub-flow state")
	}
	stance, _ := state[SubStateStance].(string)
	return map[string]any{"args": args, "stance": stance}, nil
}

func (n *EvaluateArgumentNode) Act(ctx context.Context, thought any) (any, error) {
	thoughtMap, _ := thought.(map[string]any)
	currentArgs, _ := thoughtMap["args"].([]string)
	stance, _ := thoughtMap["stance"].(string)

	log.Printf("[%s] Evaluating arguments...", stance)
	evaluationResult := callLLM(ctx, n.llmClient, "Evaluate if the following arguments are sufficient. Be extra critical regarding the quality of the arguments, and ensure that only the best possible arguments are included: %v. Respond ONLY with 'sufficient' or 'refine'.", currentArgs)

	if strings.HasPrefix(evaluationResult, "LLM_") {
		log.Printf("Warning: LLM error during evaluation: %s. Defaulting to refine.", evaluationResult)
		return "refine", nil
	}

	resp := strings.TrimSuffix(strings.ToLower(strings.TrimSpace(evaluationResult)), ".")
	if resp == "sufficient" || resp == "refine" {
		return resp, nil
	}

	log.Printf("Warning: Unexpected evaluation response from LLM: '%s'. Defaulting to refine.", evaluationResult)
	return "refine", nil
}

func (n *EvaluateArgumentNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	evaluation, ok := actResult.(string)
	if !ok {
		return "", fmt.Errorf("invalid actResult type for evaluation: %T", actResult)
	}

	iteration, _ := state[SubStateIteration].(int)
	maxIterations, _ := state[SubStateMaxIterations].(int)
	stance, _ := state[SubStateStance].(string)

	if evaluation == "sufficient" {
		log.Printf("[%s] Arguments deemed sufficient.", stance)
		return "store", nil
	} else if iteration >= maxIterations {
		log.Printf("[%s] Max iterations (%d) reached. Storing current arguments.", stance, maxIterations)
		return "store", nil
	} else {
		log.Printf("[%s] Arguments need refinement", stance)
		state[SubStateIteration] = iteration + 1
		return "refine", nil
	}
}

// RefineArgumentNode: Refines the current arguments.
type RefineArgumentNode struct {
	mote.BaseNode
	llmClient *genai.Client
}

func NewRefineArgumentNode(client *genai.Client) *RefineArgumentNode {
	return &RefineArgumentNode{BaseNode: mote.NewBaseNode(), llmClient: client}
}

func (n *RefineArgumentNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	args, ok := state[SubStateCurrentArgs].([]string)
	if !ok {
		return nil, fmt.Errorf("missing current_arguments for refinement")
	}
	stance, _ := state[SubStateStance].(string)
	iteration, _ := state[SubStateIteration].(int)
	return map[string]any{"args": args, "stance": stance, "iteration": iteration}, nil
}

func (n *RefineArgumentNode) Act(ctx context.Context, thought any) (any, error) {
	thoughtMap, _ := thought.(map[string]any)
	currentArgs, _ := thoughtMap["args"].([]string)
	stance, _ := thoughtMap["stance"].(string)
	iteration, _ := thoughtMap["iteration"].(int)

	log.Printf("[%s] Refining arguments (Iteration %d)...", stance, iteration)
	refinedResult := callLLM(ctx, n.llmClient, "Refine these arguments to make them stronger: %v", currentArgs)
	return refinedResult, nil
}

func (n *RefineArgumentNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	refinedResult, ok := actResult.(string)
	if !ok {
		return "", fmt.Errorf("invalid actResult type for refined args: %T", actResult)
	}

	// Check for errors returned by callLLM
	if strings.HasPrefix(refinedResult, "LLM_") {
		return "", fmt.Errorf("LLM error refining args: %s", refinedResult)
	}

	// Split the raw string into points for storage
	refinedPoints := strings.Split(refinedResult, "\n")
	var cleanedPoints []string
	for _, point := range refinedPoints {
		cleaned := strings.TrimSpace(point)
		if cleaned != "" {
			cleanedPoints = append(cleanedPoints, cleaned)
		}
	}
	if len(cleanedPoints) == 0 {
		cleanedPoints = []string{refinedResult}
	}

	state[SubStateCurrentArgs] = cleanedPoints // Store as []string
	stance, _ := state[SubStateStance].(string)
	iteration, _ := state[SubStateIteration].(int)
	log.Printf("[%s] Arguments refined. Iteration: %d", stance, iteration)
	return "evaluate", nil
}

// StoreResultNode: Stores the final arguments in the shared map.
type StoreResultNode struct {
	mote.BaseNode
}

func NewStoreResultNode() *StoreResultNode {
	return &StoreResultNode{BaseNode: mote.NewBaseNode()}
}

type storeArgs struct {
	stance     string
	finalArgs  []string // Expecting []string from state
	resultsMap map[string]DebateArgument
	mutex      *sync.Mutex
}

func (n *StoreResultNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	args := storeArgs{}
	var ok bool
	if args.stance, ok = state[SubStateStance].(string); !ok {
		return nil, fmt.Errorf("missing stance in sub-flow state for storing")
	}
	// Expect []string here, as set by Generate/Refine Decide methods
	if args.finalArgs, ok = state[SubStateCurrentArgs].([]string); !ok {
		log.Printf("Warning: Missing final arguments (expected []string) in sub-flow state for storing stance '%s'. Storing error marker.", args.stance)
		args.finalArgs = []string{"[Error during refinement - args missing in state]"}
	}
	if args.resultsMap, ok = state[SubStateSharedArgsMap].(map[string]DebateArgument); !ok {
		return nil, fmt.Errorf("missing shared results map reference in sub-flow state")
	}
	if args.mutex, ok = state[SubStateSharedMutex].(*sync.Mutex); !ok {
		return nil, fmt.Errorf("missing shared mutex reference in sub-flow state")
	}
	return args, nil
}

func (n *StoreResultNode) Act(ctx context.Context, thought any) (any, error) {
	// Act uses the []string directly now
	args, _ := thought.(storeArgs)
	log.Printf("[%s] Storing final arguments in shared map...", args.stance)
	args.mutex.Lock()
	args.resultsMap[args.stance] = DebateArgument{Stance: args.stance, Points: args.finalArgs}
	args.mutex.Unlock()
	log.Printf("[%s] Final arguments stored.", args.stance)
	return nil, nil
}

func (n *StoreResultNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	return "", nil // End of this sub-flow path
}

// JudgeDebateNode: Evaluates the final arguments from the shared map.
type JudgeDebateNode struct {
	mote.BaseNode
	llmClient *genai.Client
}

func NewJudgeDebateNode(client *genai.Client) *JudgeDebateNode {
	return &JudgeDebateNode{BaseNode: mote.NewBaseNode(), llmClient: client}
}

func (n *JudgeDebateNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	resultsMap, ok := state[StateFinalArgsMap].(map[string]DebateArgument)
	if !ok {
		return nil, fmt.Errorf("missing '%s' for judging", StateFinalArgsMap)
	}
	if len(resultsMap) == 0 {
		log.Println("Warning: Final arguments map is empty, cannot judge.")
		return nil, fmt.Errorf("final arguments map is empty")
	}
	return resultsMap, nil
}

func (n *JudgeDebateNode) Act(ctx context.Context, thought any) (any, error) {
	resultsMap, _ := thought.(map[string]DebateArgument)
	log.Println("MainFlow: Judging the debate based on final arguments...")

	var proArgs, conArgs []string
	for _, arg := range resultsMap {
		if strings.Contains(arg.Stance, "pro") {
			proArgs = append(proArgs, arg.Points...)
		} else {
			conArgs = append(conArgs, arg.Points...)
		}
	}

	log.Printf("Pro arguments: %v", proArgs)
	log.Printf("Con arguments: %v", conArgs)

	finalJudgement := callLLM(ctx, n.llmClient, "Evaluate the debate arguments and decide who the winner is: Pro - %v, Con - %v.", proArgs, conArgs)

	// Print the result directly
	fmt.Println("\n--- Debate Results ---")
	if strings.HasPrefix(finalJudgement, "LLM_") {
		log.Printf("Warning/Error during judgment LLM call: %s", finalJudgement)
		judgementOutput := "[Judgement unavailable due to LLM error: " + finalJudgement + "]"
		fmt.Println(judgementOutput)
		return judgementOutput, nil // Return the error marker string
	} else {
		fmt.Println(finalJudgement)
		fmt.Println("----------------------")
		return finalJudgement, nil // Return the actual judgement
	}
}

func (n *JudgeDebateNode) Decide(ctx context.Context, state mote.SharedState, actResult any) (string, error) {
	judgement, ok := actResult.(string)
	if !ok {
		// This case might happen if Act returns an error marker string
		judgement = "[Error retrieving judgement]"
		log.Printf("Warning: Invalid actResult type for judgment: %T. Storing error marker.", actResult)
	}
	state[StateFinalJudgement] = judgement
	log.Println("MainFlow: Final judgment stored. Ending flow.")
	return "", nil // End the flow here
}

// DisplayResultsNode: Prints the final judgment.
type DisplayResultsNode struct {
	mote.BaseNode
}

func NewDisplayResultsNode() *DisplayResultsNode {
	return &DisplayResultsNode{BaseNode: mote.NewBaseNode()}
}

func (n *DisplayResultsNode) Think(ctx context.Context, state mote.SharedState) (any, error) {
	judgement, ok := state[StateFinalJudgement].(string)
	if !ok {
		log.Println("Warning: Final judgment not found in state.")
		return "[No judgment available]", nil
	}
	return judgement, nil
}

func (n *DisplayResultsNode) Act(ctx context.Context, thought any) (any, error) {
	judgement, _ := thought.(string)
	fmt.Println("\n--- Debate Results ---")
	fmt.Println(judgement)
	fmt.Println("----------------------")
	return nil, nil
}

// --- Helper for Running Sub-Flows ---
func runSubFlow(ctx context.Context, flow *mote.Flow, initialState mote.SharedState, wg *sync.WaitGroup) {
	defer wg.Done()
	stance, _ := initialState[SubStateStance].(string) // For logging
	log.Printf("SubFlow [%s]: Starting execution.", stance)

	finalState, err := flow.Run(ctx, initialState)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			log.Printf("SubFlow [%s]: Execution TIMED OUT. Final state: %v", stance, finalState)
			// Ensure something is stored so the main flow doesn't hang indefinitely if Judge expects all keys
			mutex, okMutex := initialState[SubStateSharedMutex].(*sync.Mutex)
			resultsMap, okMap := initialState[SubStateSharedArgsMap].(map[string]DebateArgument)
			if okMutex && okMap {
				mutex.Lock()
				resultsMap[stance] = DebateArgument{Stance: stance, Points: []string{"[Sub-flow Timed Out]"}}
				mutex.Unlock()
			}
		} else {
			log.Printf("SubFlow [%s]: Error during execution: %v. Final state: %v", stance, err, finalState)
		}
	} else {
		log.Printf("SubFlow [%s]: Execution completed successfully.", stance)
	}
}

// --- Main Function ---
func main() {
	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey: os.Getenv("GEMINI_API_KEY"),
	})
	if err != nil {
		log.Fatalf("Failed to create Google API client: %v", err)
	}
	log.Println("Starting Parallel Debate Simulation with Pre-Built Sub-Flows...")

	// Config
	debateTopic := "Is nuclear fusion a viable energy source for the near future?"
	debateStances := []string{"pro-fusion", "anti-fusion"}
	maxSubFlowIterations := 3 // Limit refinement loops

	// --- Pre-build Sub-Flows ---
	subFlows := make(map[string]*mote.Flow)
	for _, stance := range debateStances {
		generateNode := NewGenerateInitialArgumentNode(client)
		evaluateNode := NewEvaluateArgumentNode(client)
		refineNode := NewRefineArgumentNode(client)
		storeNode := NewStoreResultNode()

		generateNode.AddSuccessor("evaluate", evaluateNode)
		evaluateNode.AddSuccessor("refine", refineNode)
		evaluateNode.AddSuccessor("store", storeNode)
		refineNode.AddSuccessor("evaluate", evaluateNode)

		// Create and store the sub-flow instance
		subFlows[stance] = mote.NewFlow(generateNode)
	}
	log.Println("Main: All sub-flows defined.")

	// --- Prepare Shared State ---
	// No main flow nodes left before judging, prepare state directly
	mainInitialState := mote.SharedState{
		StateTopic:        debateTopic,
		StateStances:      debateStances,
		StateFinalArgsMap: make(map[string]DebateArgument),
		StateMutex:        &sync.Mutex{},
		StateWaitGroup:    &sync.WaitGroup{},
		StateSubFlowsMap:  subFlows,
	}

	// --- Directly Launch and Manage Sub-Flows in Main ---
	log.Printf("Main: Launching %d pre-built sub-flows in parallel...", len(debateStances))

	// Retrieve necessary components from the state map
	topic := mainInitialState[StateTopic].(string)
	stances := mainInitialState[StateStances].([]string)
	subFlowsMap := mainInitialState[StateSubFlowsMap].(map[string]*mote.Flow)
	resultsMap := mainInitialState[StateFinalArgsMap].(map[string]DebateArgument)
	mutex := mainInitialState[StateMutex].(*sync.Mutex)
	wg := mainInitialState[StateWaitGroup].(*sync.WaitGroup)

	for _, stance := range stances {
		subFlow, exists := subFlowsMap[stance]
		if !exists {
			log.Printf("Error: Sub-flow for stance '%s' not found in map! Skipping.", stance)
			continue
		}

		// Create initial state for this specific sub-flow run
		subInitialState := mote.SharedState{
			SubStateTopic:         topic,
			SubStateStance:        stance,
			SubStateSharedArgsMap: resultsMap,
			SubStateSharedMutex:   mutex,
			SubStateMaxIterations: maxSubFlowIterations, // Get max iterations from main scope
		}

		// Launch the sub-flow in a goroutine
		wg.Add(1)
		go runSubFlow(context.Background(), subFlow, subInitialState, wg)
	}

	log.Println("Main: Waiting for sub-flows to complete...")
	wg.Wait()
	log.Println("Main: All sub-flows finished.")

	// --- Run Final Judging Step ---
	judgeNode := NewJudgeDebateNode(client)
	judgingFlow := mote.NewFlow(judgeNode)

	log.Println("Main: Starting final judgment flow...")
	_, err = judgingFlow.Run(context.Background(), mainInitialState) // Run the judging flow

	if err != nil {
		log.Fatalf("Main: Judging flow execution failed: %v", err)
	} else {
		log.Println("Main: Judging flow completed successfully.")
	}
}
