package mote

import (
	"context"
	"fmt"
	"log"
)

// SharedState represents the mutable context shared between nodes in a flow.
// It uses a map for flexibility, allowing nodes to pass arbitrary data.
type SharedState map[string]any

// Node represents a single step in an execution graph (an agent's capability).
type Node interface {
	// Think analyzes the current SharedState to prepare for action.
	// It extracts necessary data and returns it as 'thought' (input for Act).
	// It may modify state (e.g., consuming items from a queue).
	Think(ctx context.Context, state SharedState) (thought any, err error)

	// Act performs the core logic or interaction based on the 'thought'.
	// It should avoid direct modification of SharedState.
	// Returns the result of the action ('actResult').
	Act(ctx context.Context, thought any) (actResult any, err error)

	// Decide processes the 'actResult', updates SharedState, and determines
	// the next step by returning an 'action' string.
	Decide(ctx context.Context, state SharedState, actResult any) (action string, err error)

	// GetSuccessors returns the map of action strings to successor nodes.
	GetSuccessors() map[string]Node

	// AddSuccessor links another node as a successor for a given action.
	// Returns the successor node to allow chaining definitions (e.g., nodeA.AddSuccessor("default", nodeB).AddSuccessor("done", nodeC)).
	AddSuccessor(action string, successor Node) Node
}

// BaseNode provides default implementations and successor handling.
// Concrete node implementations should embed this struct.
type BaseNode struct {
	successors map[string]Node
}

// NewBaseNode initializes a BaseNode, typically called by embedding structs.
func NewBaseNode() BaseNode {
	return BaseNode{
		successors: make(map[string]Node),
	}
}

// Think (Default): Does nothing, returns nil thought. Assumes Act needs no input.
func (b *BaseNode) Think(ctx context.Context, state SharedState) (any, error) {
	return nil, nil
}

// Act (Default): Does nothing, returns nil result. Assumes the node is primarily for state manipulation or flow control.
func (b *BaseNode) Act(ctx context.Context, thought any) (any, error) {
	return nil, nil
}

// Decide (Default): Does nothing to state, returns "default" action. Assumes a single, unconditional next step.
func (b *BaseNode) Decide(ctx context.Context, state SharedState, actResult any) (string, error) {
	return "default", nil
}

// GetSuccessors returns the successor map. Required by the Node interface.
func (b *BaseNode) GetSuccessors() map[string]Node {
	// Return a copy to prevent external modification? For now, return original.
	return b.successors
}

// AddSuccessor adds or updates a successor for a given action.
// Implements the Node interface requirement and enables chaining.
func (b *BaseNode) AddSuccessor(action string, successor Node) Node {
	if _, exists := b.successors[action]; exists {
		// Using standard logger for warnings. Consider a more structured logger.
		log.Printf("Warning: Overwriting successor for action '%s' on node (type %T)", action, b)
	}
	b.successors[action] = successor
	return successor // Enable chaining
}

// --- Flow ---

// Flow orchestrates the execution of a graph of connected Nodes.
type Flow struct {
	startNode Node
}

// NewFlow creates a new Flow with a designated starting Node.
func NewFlow(start Node) *Flow {
	if start == nil {
		log.Println("Warning: Creating a Flow with a nil start node. Run will fail.")
	}
	return &Flow{startNode: start}
}

// Run executes the flow sequentially starting from the startNode.
// It manages the Think -> Act -> Decide lifecycle for each node.
// Takes an initial state and returns the final state and any error encountered.
// Execution stops if a node returns an error, context is cancelled, or
// a node's decided action has no corresponding successor.
func (f *Flow) Run(ctx context.Context, initialState SharedState) (finalState SharedState, err error) {
	if f.startNode == nil {
		return initialState, fmt.Errorf("flow cannot run: start node is nil")
	}

	currentNode := f.startNode
	state := initialState
	if state == nil {
		state = make(SharedState)
	}

	// Defer error handling to wrap errors with node context
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("panic during flow execution: %v", r)
		}
		if err != nil && currentNode != nil {
			err = fmt.Errorf("error at node %T: %w", currentNode, err)
		}
	}()

	for currentNode != nil {
		// Check for context cancellation before each node execution
		select {
		case <-ctx.Done():
			log.Printf("Info: Flow execution cancelled by context at node %T.", currentNode)
			return state, ctx.Err()
		default:
			// Continue execution
		}

		// Think Phase
		thought, thinkErr := currentNode.Think(ctx, state)
		if thinkErr != nil {
			err = thinkErr
			return state, err
		}

		// Act Phase
		actResult, actErr := currentNode.Act(ctx, thought)
		if actErr != nil {
			// TODO: Implement node-level retry/fallback logic here if desired.
			err = actErr
			return state, err
		}

		action, decideErr := currentNode.Decide(ctx, state, actResult)
		if decideErr != nil {
			err = decideErr
			return state, err
		}

		// Transition Logic
		successors := currentNode.GetSuccessors()
		nextNode, ok := successors[action]

		if !ok {
			// Specific action not found, try "default" if the action wasn't already "default"
			if action != "default" {
				nextNode, ok = successors["default"]
			}
		}

		if !ok {
			// No specific successor and no default (or action was already default).
			// This is a graceful termination condition for this path.
			if len(successors) > 0 {
				log.Printf("Info: Flow path ending: Action '%s' from node %T has no matching successor.", action, currentNode)
			} else {
				log.Printf("Info: Flow path ending: Node %T has no successors.", currentNode)
			}
			currentNode = nil
		} else {
			currentNode = nextNode
		}
	}

	log.Println("Info: Flow execution completed successfully.")
	return state, nil
}
