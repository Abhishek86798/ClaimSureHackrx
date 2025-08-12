#!/usr/bin/env python3
"""
Demo script for enhanced GPT-3.5 integration with Claimsure.

This script demonstrates:
1. GPT-3.5 service initialization and status
2. Different processing strategies
3. Fallback mechanisms
4. Query type classification
5. Performance metrics
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.hybrid_processor import HybridProcessor, ProcessingStrategy
from core.gpt35_service import GPT35Service, GPT35Status

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpt35_service():
    """Test the GPT-3.5 service directly"""
    print("🔧 Testing GPT-3.5 Service")
    print("=" * 50)
    
    # Initialize service
    service = GPT35Service()
    
    # Show service status
    status = service.get_service_status()
    print(f"Service Status: {status['status']}")
    print(f"Model: {status['model']}")
    print(f"API Key Configured: {status['api_key_configured']}")
    print(f"Client Available: {status['client_available']}")
    
    if service.status == GPT35Status.AVAILABLE:
        print("✅ GPT-3.5 service is available!")
        
        # Test with sample insurance query
        test_query = "What is covered under my health insurance policy for emergency room visits?"
        test_context = """
        Your health insurance policy covers:
        - Emergency room visits with $100 copay
        - Hospital stays with 80% coverage after $500 deductible
        - Doctor visits with $25 copay
        - Prescription medications with tiered pricing
        - Preventive care at 100% coverage
        
        Exclusions:
        - Cosmetic procedures
        - Experimental treatments
        - Dental and vision (separate coverage)
        - Out-of-network providers (except emergencies)
        """
        
        print(f"\n📝 Testing with query: {test_query}")
        print(f"   Query type: Emergency coverage inquiry")
        print(f"   Context length: {len(test_context)} characters")
        
        # Process query
        start_time = time.time()
        result = service.process_insurance_query(test_query, test_context, "coverage")
        total_time = time.time() - start_time
        
        print(f"\n📊 Results:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Tokens Used: {result.tokens_used}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Model: {result.model_used}")
        
        if result.error_message:
            print(f"   Error: {result.error_message}")
        else:
            print(f"\n💬 Response:")
            print(f"   {result.content}")
            
    else:
        print(f"❌ Service not available: {service.status.value}")
        if service.status == GPT35Status.UNAVAILABLE:
            print("   Please check your OPENAI_API_KEY environment variable")
        elif service.status == GPT35Status.RATE_LIMITED:
            print("   Service is currently rate limited. Please wait and try again.")
        elif service.status == GPT35Status.QUOTA_EXCEEDED:
            print("   API quota exceeded. Please check your OpenAI account.")
    
    return service

def test_hybrid_processor():
    """Test the hybrid processor with GPT-3.5 integration"""
    print("\n🔧 Testing Hybrid Processor with GPT-3.5")
    print("=" * 50)
    
    # Initialize hybrid processor
    processor = HybridProcessor()
    
    # Show processor statistics
    stats = processor.get_processing_statistics()
    print(f"Confidence Threshold: {stats['confidence_threshold']}")
    print(f"Complexity Threshold: {stats['complexity_threshold']}")
    print(f"Available Processors:")
    for name, available in stats['available_processors'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {name}")
    
    # Test different query types
    test_queries = [
        {
            "query": "What is covered under my health insurance?",
            "type": "coverage",
            "complexity": "low"
        },
        {
            "query": "How do I file a claim for a hospital stay and what documents do I need to submit?",
            "type": "claims",
            "complexity": "high"
        },
        {
            "query": "What are the coverage limits and deductibles for prescription medications?",
            "type": "limits",
            "complexity": "medium"
        },
        {
            "query": "Can you explain the difference between in-network and out-of-network coverage?",
            "type": "general",
            "complexity": "medium"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test Case {i}: {test_case['type'].title()} Query")
        print(f"   Complexity: {test_case['complexity']}")
        print(f"   Query: {test_case['query']}")
        
        # Process query
        start_time = time.time()
        result = processor.process_query(test_case['query'])
        total_time = time.time() - start_time
        
        print(f"\n📊 Results:")
        print(f"   Strategy Used: {result.strategy_used.value}")
        print(f"   Answer: {result.answer[:100]}...")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Processing Time: {result.processing_time:.3f}s")
        print(f"   Total Time: {total_time:.3f}s")
        print(f"   Source Clauses: {len(result.source_clauses)}")
        
        # Show metadata
        if result.metadata:
            print(f"   Metadata:")
            for key, value in result.metadata.items():
                if key != "strategy":  # Already shown above
                    print(f"     {key}: {value}")
    
    return processor

def test_fallback_scenarios():
    """Test fallback scenarios when GPT-3.5 is unavailable"""
    print("\n🔧 Testing Fallback Scenarios")
    print("=" * 50)
    
    # Test with different strategies
    processor = HybridProcessor()
    
    # Test forced strategies
    test_query = "What is covered under my health insurance policy?"
    
    strategies = [
        ProcessingStrategy.LOCAL_ONLY,
        ProcessingStrategy.FREE_API,
        ProcessingStrategy.HYBRID,
        ProcessingStrategy.GPT35
    ]
    
    for strategy in strategies:
        print(f"\n📝 Testing {strategy.value.upper()} strategy:")
        
        try:
            start_time = time.time()
            result = processor.process_query(test_query, force_strategy=strategy)
            total_time = time.time() - start_time
            
            print(f"   Strategy: {result.strategy_used.value}")
            print(f"   Answer: {result.answer[:80]}...")
            print(f"   Confidence: {result.confidence:.3f}")
            print(f"   Processing Time: {result.processing_time:.3f}s")
            print(f"   Total Time: {total_time:.3f}s")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_batch_processing():
    """Test batch processing capabilities"""
    print("\n🔧 Testing Batch Processing")
    print("=" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No API key available for batch processing test")
        return
    
    service = GPT35Service()
    
    if service.status != GPT35Status.AVAILABLE:
        print(f"❌ Service not available: {service.status.value}")
        return
    
    # Prepare batch queries
    batch_queries = [
        ("What is covered under health insurance?", "Health insurance covers doctor visits, hospital stays, and prescription drugs.", "coverage"),
        ("How do I file a claim?", "To file a claim, submit the claim form with supporting documents within 30 days.", "claims"),
        ("What are the coverage limits?", "Coverage limits vary by plan type and service category.", "limits")
    ]
    
    print(f"Processing {len(batch_queries)} queries in batch...")
    
    start_time = time.time()
    results = service.batch_process_queries(batch_queries)
    total_time = time.time() - start_time
    
    print(f"Batch processing completed in {total_time:.2f}s")
    
    for i, result in enumerate(results, 1):
        print(f"\n📊 Query {i} Results:")
        print(f"   Status: {result.status.value}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Tokens: {result.tokens_used}")
        print(f"   Time: {result.processing_time:.3f}s")
        if result.error_message:
            print(f"   Error: {result.error_message}")

def main():
    """Main demo function"""
    print("🚀 Claimsure GPT-3.5 Integration Demo")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY environment variable found")
        print("   Please set your OpenAI API key to test GPT-3.5 functionality")
        print("   You can still test the fallback mechanisms")
        print()
    
    try:
        # Test GPT-3.5 service
        gpt35_service = test_gpt35_service()
        
        # Test hybrid processor
        hybrid_processor = test_hybrid_processor()
        
        # Test fallback scenarios
        test_fallback_scenarios()
        
        # Test batch processing if available
        if api_key:
            test_batch_processing()
        
        print("\n✅ Demo completed successfully!")
        print("\n📋 Summary:")
        print("   - GPT-3.5 service integrated with hybrid processor")
        print("   - Intelligent strategy selection based on query complexity")
        print("   - Robust fallback mechanisms for reliability")
        print("   - Query type classification for better prompting")
        print("   - Performance monitoring and error handling")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo execution failed")

if __name__ == "__main__":
    main()

