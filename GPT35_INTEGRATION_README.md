# 🚀 GPT-3.5 Integration for Claimsure

This document explains how to use the enhanced GPT-3.5 integration with your Claimsure insurance document processing system.

## ✨ Features

- **Intelligent Strategy Selection**: Automatically chooses between GPT-3.5, local processing, and hybrid approaches based on query complexity
- **Free Tier Optimization**: Designed specifically for GPT-3.5 free tier usage with cost controls
- **Robust Fallbacks**: Seamlessly falls back to local processing when GPT-3.5 is unavailable
- **Insurance-Specific Prompting**: Tailored prompts for different types of insurance queries
- **Performance Monitoring**: Track usage, costs, and performance metrics
- **Rate Limiting**: Built-in rate limiting and retry logic for reliable operation

## 🛠️ Setup

### 1. Install Dependencies

The required dependencies are already in your `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY="your-api-key-here"

# Windows Command Prompt
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in your project root:

```env
OPENAI_API_KEY=your-api-key-here
```

### 3. Verify Installation

Run the test script to verify everything is working:

```bash
python test_gpt35_integration.py
```

## 🔧 Configuration

### Basic Configuration

The system uses `gpt35_config.py` for all settings. Key configurations:

```python
# Model settings
GPT35_MODEL = "gpt-3.5-turbo"  # Best for free tier
GPT35_MAX_TOKENS = 1000        # Conservative limit
GPT35_TEMPERATURE = 0.1        # Consistent responses

# Free tier management
GPT35_FREE_TIER_ENABLED = True
GPT35_TOKEN_BUDGET = 1000      # Daily token budget

# Fallback behavior
GPT35_FALLBACK_ENABLED = True
GPT35_CONFIDENCE_THRESHOLD = 0.6
```

### Environment Variable Overrides

You can override any setting using environment variables:

```bash
export GPT35_MAX_TOKENS=500
export GPT35_TEMPERATURE=0.2
export GPT35_FREE_TIER_ENABLED=false
```

## 🚀 Usage

### Basic Usage

```python
from src.core.hybrid_processor import HybridProcessor

# Initialize processor (automatically includes GPT-3.5)
processor = HybridProcessor()

# Process a query
result = processor.process_query("What is covered under my health insurance?")

print(f"Answer: {result.answer}")
print(f"Strategy: {result.strategy_used.value}")
print(f"Confidence: {result.confidence}")
```

### Force Specific Strategy

```python
from src.core.hybrid_processor import ProcessingStrategy

# Force GPT-3.5 processing
result = processor.process_query(
    "Explain my coverage limits in detail",
    force_strategy=ProcessingStrategy.GPT35
)
```

### Direct GPT-3.5 Service Usage

```python
from src.core.gpt35_service import GPT35Service

service = GPT35Service()

# Process insurance query
result = service.process_insurance_query(
    query="What is covered under my policy?",
    context="Your policy covers...",
    query_type="coverage"
)

if result.status.value == "available":
    print(f"Answer: {result.content}")
    print(f"Confidence: {result.confidence}")
    print(f"Tokens used: {result.tokens_used}")
```

## 📊 Query Types

The system automatically classifies queries into different types for better prompting:

- **Coverage**: What is/isn't covered under the policy
- **Claims**: How to file claims and required procedures
- **Limits**: Coverage limits, deductibles, and maximums
- **Costs**: Pricing, fees, and payment information
- **General**: General policy explanations and guidance

## 🔄 Processing Strategies

The hybrid processor automatically selects the best strategy:

1. **GPT-3.5**: For complex queries when service is available
2. **Hybrid**: Combines local and API processing for medium complexity
3. **Local Only**: For simple queries or when external services fail
4. **Free API**: For complex queries when GPT-3.5 is unavailable
5. **Fallback**: Emergency fallback when all else fails

## 📈 Performance Monitoring

### Service Status

```python
service = GPT35Service()
status = service.get_service_status()

print(f"Status: {status['status']}")
print(f"Model: {status['model']}")
print(f"API Key: {'Configured' if status['api_key_configured'] else 'Missing'}")
```

### Processing Statistics

```python
processor = HybridProcessor()
stats = processor.get_processing_statistics()

print(f"Available Processors:")
for name, available in stats['available_processors'].items():
    print(f"  {name}: {'✅' if available else '❌'}")
```

## 🧪 Testing

### Run Basic Tests

```bash
python test_gpt35_integration.py
```

### Run Full Demo

```bash
python demo_gpt35_integration.py
```

### Test Without API Key

The system gracefully handles missing API keys:

```python
# This will work even without an API key
processor = HybridProcessor()
result = processor.process_query("What is covered?")

# Will automatically use local processing
print(f"Strategy used: {result.strategy_used.value}")
```

## 💰 Cost Management

### Free Tier Optimization

- **Token Budget**: Set daily token limits
- **Conservative Limits**: Default 1000 token limit per request
- **Smart Fallbacks**: Use local processing when possible
- **Cost Tracking**: Monitor token usage and costs

### Cost Calculation

```python
# Estimate costs
tokens_used = 500
cost_per_1k = 0.002  # GPT-3.5-turbo pricing
estimated_cost = (tokens_used / 1000) * cost_per_1k
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## 🚨 Error Handling

### Common Error Scenarios

1. **No API Key**: Automatically falls back to local processing
2. **Rate Limited**: Implements exponential backoff and retries
3. **Quota Exceeded**: Gracefully degrades to fallback processing
4. **Service Unavailable**: Uses local processing with informative messages

### Error Messages

```python
# Customizable error messages in gpt35_config.py
GPT35_ERROR_MESSAGES = {
    "rate_limit": "Service is temporarily busy. Please try again in a moment.",
    "quota_exceeded": "Service quota exceeded. Please try again later.",
    "service_unavailable": "Service is temporarily unavailable. Using fallback processing."
}
```

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src` is in your Python path
2. **API Key Issues**: Check environment variable or `.env` file
3. **Rate Limiting**: Increase delays in configuration
4. **Memory Issues**: Reduce `max_tokens` in configuration

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Examples

### Insurance Coverage Query

```python
query = "What is covered under my health insurance for emergency room visits?"
result = processor.process_query(query)

# System automatically:
# 1. Classifies as "coverage" query type
# 2. Selects GPT-3.5 strategy (if available)
# 3. Uses insurance-specific prompting
# 4. Falls back to local processing if needed
```

### Claims Process Query

```python
query = "How do I file a claim for a hospital stay and what documents do I need?"
result = processor.process_query(query)

# System automatically:
# 1. Classifies as "claims" query type
# 2. Uses claims-specific prompting
# 3. Provides step-by-step guidance
```

## 🔮 Future Enhancements

- **Multi-Model Support**: Add support for other LLM providers
- **Advanced Prompting**: Dynamic prompt generation based on document content
- **Cost Optimization**: Intelligent token usage optimization
- **Performance Analytics**: Detailed performance metrics and insights
- **Custom Models**: Support for fine-tuned insurance-specific models

## 📞 Support

If you encounter issues:

1. Check the test scripts first
2. Verify your API key configuration
3. Check the logs for detailed error messages
4. Ensure all dependencies are installed correctly

## 🎯 Best Practices

1. **Start Simple**: Begin with basic queries to test the system
2. **Monitor Costs**: Keep track of token usage, especially on free tier
3. **Use Fallbacks**: Let the system automatically choose the best strategy
4. **Test Fallbacks**: Verify local processing works when GPT-3.5 is unavailable
5. **Customize Prompts**: Adjust insurance-specific prompts for your use case

---

**Happy coding! 🚀**

Your Claimsure system now has enterprise-grade GPT-3.5 integration with intelligent fallbacks and cost optimization!

