"""callreplay: regression tests for voice and chat agents, built from recorded conversations."""
from .contract import Contract
from .contract import load as load_contract
from .evaluate import Result, evaluate
from .formats import load as load_conversations
from .model import Conversation, ToolCall, Turn
from .replay import Replayed, replay_all, replay_one

__version__ = '0.1.0'
__all__ = ['Contract', 'Conversation', 'Replayed', 'Result', 'ToolCall', 'Turn', 'evaluate', 'load_contract',
           'load_conversations', 'replay_all', 'replay_one', '__version__']
