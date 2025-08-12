#!/usr/bin/env python3
"""
Quick Test for Enhanced HF Service
"""

import sys
sys.path.insert(0, 'src')

def test_basic_import():
    """Test basic imports."""
    print("🔧 Testing Basic Imports...")
    
    try:
        from core.hf_enhanced_service import HFEnhancedService
        print("  ✅ HF Enhanced Service imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_service_initialization():
    """Test service initialization."""
    print("\n🔧 Testing Service Initialization...")
    
    try:
        from core.hf_enhanced_service import HFEnhancedService
        
        service = HFEnhancedService()
        print("  ✅ Service initialized")
        
        status = service.get_model_status()
        print("  📊 Model Status:")
        for model, available in status.items():
            status_icon = "✅" if available else "❌"
            print(f"    {status_icon} {model}")
        
        return True
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        return False

def test_hybrid_processor():
    """Test hybrid processor integration."""
    print("\n🔧 Testing Hybrid Processor...")
    
    try:
        from core.hybrid_processor import HybridProcessor, ProcessingStrategy
        
        processor = HybridProcessor()
        print("  ✅ Hybrid Processor initialized")
        
        # Test strategy selection
        test_query = "What is the waiting period for pre-existing diseases?"
        strategy = processor._select_strategy(test_query)
        print(f"  ✅ Selected strategy: {strategy.value}")
        
        return True
    except Exception as e:
        print(f"  ❌ Hybrid processor test failed: {e}")
        return False

def main():
    """Run quick tests."""
    print("🏥 Quick Enhanced HF Test")
    print("=" * 30)
    
    tests = [
        ("Basic Import", test_basic_import),
        ("Service Initialization", test_service_initialization),
        ("Hybrid Processor", test_hybrid_processor),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
    
    print(f"\n🎯 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All quick tests passed!")
    else:
        print("⚠️ Some tests failed.")
    
    return passed == len(tests)

if __name__ == "__main__":
    main()
