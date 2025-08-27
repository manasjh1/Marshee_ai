from pymongo.collection import Collection
from modals.chat import ChatSession, ChatMessage

class ChatRepository:
    """
    Handles all database operations related to chat sessions and messages.
    """
    def __init__(self, sessions_collection: Collection, messages_collection: Collection):
        """
        Initializes the repository with specific MongoDB collections.

        Args:
            sessions_collection: The PyMongo collection for chat sessions.
            messages_collection: The PyMongo collection for chat messages.
        """
        self.sessions_collection = sessions_collection
        self.messages_collection = messages_collection

    async def create_session(self, session: ChatSession):
        """Creates a new chat session in the database."""
        await self.sessions_collection.insert_one(session.dict())

    async def get_session(self, session_id: str) -> ChatSession | None:
        """Retrieves a chat session by its ID."""
        session_data = await self.sessions_collection.find_one({"session_id": session_id})
        return ChatSession(**session_data) if session_data else None

    async def update_session(self, session: ChatSession):
        """Updates an existing chat session."""
        await self.sessions_collection.update_one(
            {"session_id": session.session_id},
            {"$set": session.dict()}
        )

    async def save_message(self, message: ChatMessage):
        """Saves a chat message to the database."""
        await self.messages_collection.insert_one(message.dict())

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        """Retrieves all messages for a given session."""
        messages_cursor = self.messages_collection.find({"session_id": session_id}).sort("timestamp")
        return [ChatMessage(**msg) async for msg in messages_cursor]
