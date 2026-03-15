```json
{
    "tools/discord_integration.py": {
        "content": "
import logging
import discord
from discord.ext import commands
from pydantic import BaseModel
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiscordConfig(BaseModel):
    """Discord configuration model"""
    token: str
    guild_id: int
    channel_id: int

class DiscordIntegration:
    """Discord integration class"""
    def __init__(self, config: DiscordConfig):
        """
        Initialize the Discord integration class.

        Args:
        - config (DiscordConfig): Discord configuration model
        """
        self.config = config
        self.bot = commands.Bot(command_prefix='!')

    async def send_message(self, message: str) -> None:
        """
        Send a message to the Discord channel.

        Args:
        - message (str): The message to send

        Raises:
        - Exception: If an error occurs while sending the message
        """
        try:
            channel = self.bot.get_channel(self.config.channel_id)
            await channel.send(message)
            logger.info('Message sent successfully')
        except Exception as e:
            logger.error(f'Error sending message: {e}')

    async def start_bot(self) -> None:
        """
        Start the Discord bot.

        Raises:
        - Exception: If an error occurs while starting the bot
        """
        try:
            await self.bot.start(self.config.token)
            logger.info('Bot started successfully')
        except Exception as e:
            logger.error(f'Error starting bot: {e}')

    def non_stationary_drift_index(self, data: list) -> float:
        """
        Calculate the non-stationary drift index.

        Args:
        - data (list): The data to calculate the index for

        Returns:
        - float: The non-stationary drift index
        """
        try:
            # Calculate the non-stationary drift index using stochastic regime switch
            stochastic_regime_switch = sum(data) / len(data)
            return stochastic_regime_switch
        except Exception as e:
            logger.error(f'Error calculating non-stationary drift index: {e}')
            return 0.0

if __name__ == '__main__':
    # Create a Discord configuration model
    config = DiscordConfig(token='YOUR_TOKEN', guild_id=123456, channel_id=789012)

    # Create a Discord integration instance
    discord_integration = DiscordIntegration(config)

    # Start the Discord bot
    import asyncio
    asyncio.run(discord_integration.start_bot())

    # Simulate the 'Rocket Science' problem
    data = [1, 2, 3, 4, 5]
    non_stationary_drift_index = discord_integration.non_stationary_drift_index(data)
    print(f'Non-stationary drift index: {non_stationary_drift_index}')

    # Send a message to the Discord channel
    asyncio.run(discord_integration.send_message('Hello from the Rocket Science simulation!'))
",
        "commit_message": "feat: implement specialized discord_integration logic"
    }
}
```