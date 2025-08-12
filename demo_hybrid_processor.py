#!/usr/bin/env python3
"""
Demo: Hybrid Query Processor
Showcases intelligent routing between local processing and free API models
"""

import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import (
    HybridProcessor, 
    ProcessingStrategy, 
    EmbeddingSystem,
    load_pdf,
    semantic_chunk
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_sample_data():
    """Set up sample data for testing"""
    print("🔧 Setting up sample data...")
    
    # Create sample insurance policy text
    sample_policy = """
    INSURANCE POLICY TERMS AND CONDITIONS
    
    SECTION 1: COVERAGE
    This policy provides coverage for medical expenses incurred due to accidents or illnesses.
    Coverage includes hospitalization, surgery, medication, and rehabilitation services.
    
    SECTION 2: EXCLUSIONS
    The following are not covered under this policy:
    - Pre-existing conditions diagnosed before policy start date
    - Cosmetic procedures not medically necessary
    - Experimental treatments not approved by FDA
    - Injuries sustained while participating in extreme sports
    
    SECTION 3: LIMITS AND DEDUCTIBLES
    Annual coverage limit: $100,000
    Deductible: $1,000 per year
    Co-pay: 20% after deductible is met
    
    SECTION 4: CLAIMS PROCESSING
    Claims must be submitted within 90 days of service date.
    Supporting documentation must include:
    - Medical bills and receipts
    - Doctor's diagnosis and treatment plan
    - Proof of payment
    
    SECTION 5: APPEALS PROCESS
    If a claim is denied, you may appeal within 30 days.
    Appeals will be reviewed by an independent medical review board.
    """
    
    # Create sample chunks
    chunks = semantic_chunk(sample_policy, chunk_size=300, overlap=50)
    
    print(f"✅ Created {len(chunks)} sample chunks")
    return chunks

def test_hybrid_processor():
    """Test the hybrid processor with different query types"""
    print("\n🚀 Testing Hybrid Query Processor")
    print("=" * 50)
    
    # Initialize hybrid processor
    processor = HybridProcessor(
        confidence_threshold=0.7,
        complexity_threshold=0.6
    )
    
    # Test queries of varying complexity
    test_queries = [
        # Simple query - should use LOCAL_ONLY
        "What is the annual coverage limit?",
        
        # Medium complexity - should use HYBRID
        "What documents do I need for claims?",
        
        # Complex query - should try FREE_API first
        "Can you explain why pre-existing conditions are excluded and what happens if I need treatment for a condition I had before getting the policy?",
        
        # Very complex - should use FREE_API
        "I was injured while playing basketball, which is not an extreme sport. The injury required surgery and rehabilitation. How much will be covered under my policy, and what is the process for filing a claim? Please explain the deductible and co-pay calculations.",
        
        # Simple but specific - should use LOCAL_ONLY
        "What is the deductible amount?"
    ]
    
    # Process each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process with automatic strategy selection
            result = processor.process_query(query)
            
            print(f"🎯 Strategy Used: {result.strategy_used.value}")
            print(f"⏱️  Processing Time: {result.processing_time:.3f}s")
            print(f"🎯 Confidence: {result.confidence:.2f}")
            print(f"📊 Model Used: {result.metadata['model_used']}")
            print(f"🔍 Clauses Retrieved: {result.metadata['clauses_retrieved']}")
            print(f"🧠 Query Complexity: {result.metadata['query_complexity']:.2f}")
            print(f"💡 Answer: {result.answer[:200]}{'...' if len(result.answer) > 200 else ''}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
    
    return processor

def test_strategy_forcing():
    """Test forcing specific strategies"""
    print("\n🔧 Testing Strategy Forcing")
    print("=" * 50)
    
    processor = HybridProcessor()
    query = "What is covered under this policy?"
    
    strategies = [
        ProcessingStrategy.LOCAL_ONLY,
        ProcessingStrategy.FREE_API,
        ProcessingStrategy.HYBRID,
        ProcessingStrategy.FALLBACK
    ]
    
    for strategy in strategies:
        print(f"\n🔄 Forcing Strategy: {strategy.value}")
        print("-" * 30)
        
        try:
            result = processor.process_query(query, force_strategy=strategy)
            print(f"✅ Success: {result.strategy_used.value}")
            print(f"⏱️  Time: {result.processing_time:.3f}s")
            print(f"🎯 Confidence: {result.confidence:.2f}")
        except Exception as e:
            print(f"❌ Error: {e}")

def test_with_embedding_system():
    """Test hybrid processor with actual embedding system"""
    print("\n🧠 Testing with Embedding System")
    print("=" * 50)
    
    try:
        # Initialize embedding system
        embedding_system = EmbeddingSystem()
        
        # Set up sample data
        chunks = setup_sample_data()
        
        # Add chunks to embedding system
        print("📚 Adding chunks to embedding system...")
        embedding_system.add_chunks(chunks)
        
        # Test complex query with embedding system
        complex_query = """
        I have a chronic condition that was diagnosed 2 years ago, but I'm considering 
        switching to this new policy. The condition requires ongoing medication and 
        occasional specialist visits. What should I know about coverage and exclusions?
        """
        
        print(f"\n📝 Complex Query: {complex_query.strip()}")
        print("-" * 50)
        
        processor = HybridProcessor()
        result = processor.process_query(complex_query, embedding_system)
        
        print(f"🎯 Strategy: {result.strategy_used.value}")
        print(f"⏱️  Time: {result.processing_time:.3f}s")
        print(f"🎯 Confidence: {result.confidence:.2f}")
        print(f"🔍 Clauses: {result.metadata['clauses_retrieved']}")
        print(f"💡 Answer: {result.answer}")
        
        if result.source_clauses:
            print(f"\n📋 Supporting Clauses:")
            for i, clause in enumerate(result.source_clauses[:3], 1):
                print(f"  {i}. {clause.get('text', '')[:100]}...")
        
    except Exception as e:
        print(f"❌ Error testing with embedding system: {e}")

def show_processing_statistics():
    """Show processing statistics and capabilities"""
    print("\n📊 Processing Statistics")
    print("=" * 50)
    
    processor = HybridProcessor()
    stats = processor.get_processing_statistics()
    
    print(f"🎯 Confidence Threshold: {stats['confidence_threshold']}")
    print(f"🧠 Complexity Threshold: {stats['complexity_threshold']}")
    print(f"⚖️  Strategy Weights:")
    for strategy, weight in stats['strategy_weights'].items():
        print(f"    {strategy}: {weight}")
    
    print(f"\n🔧 Available Processors:")
    for processor_name, available in stats['available_processors'].items():
        status = "✅ Available" if available else "❌ Not Available"
        print(f"    {processor_name}: {status}")

def main():
    """Main demo function"""
    print("🌟 Hybrid Query Processor Demo")
    print("=" * 60)
    print("This demo showcases intelligent routing between local processing")
    print("and free API models based on query complexity and confidence.")
    print("=" * 60)
    
    try:
        # Test basic functionality
        processor = test_hybrid_processor()
        
        # Test strategy forcing
        test_strategy_forcing()
        
        # Test with embedding system
        test_with_embedding_system()
        
        # Show statistics
        show_processing_statistics()
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Key Benefits of Hybrid Approach:")
        print("   • Local processing for simple queries (fast & free)")
        print("   • API enhancement for complex reasoning")
        print("   • Automatic strategy selection based on complexity")
        print("   • Graceful fallback when APIs are unavailable")
        print("   • Cost optimization through intelligent routing")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.exception("Demo error")

if __name__ == "__main__":
    main()
