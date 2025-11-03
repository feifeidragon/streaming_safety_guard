# run_interactive.py

import os
import sys
import yaml
import torch
from loguru import logger
from core.streaming_guard import StreamingGuard


def print_banner():
    """打印系统横幅"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║     AI Content Streaming Safety Guard System v1.0         ║
║     实时流式生成内容安全拦截系统                            ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证必需的配置项
        model_path = config.get('model', {}).get('local_model_path')
        if not model_path:
            logger.error("local_model_path not found in config!")
            logger.info("Please set model.local_model_path in config/config.yaml")
            sys.exit(1)

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"Model path does not exist: {model_path}")
            sys.exit(1)

        # 合并配置，适配StreamingGuard的参数
        merged_config = {
            # 模型配置
            'model_path': model_path,  # StreamingGuard使用model_path
            'device': config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu'),
            'use_fp16': config['model'].get('torch_dtype', 'float16') == 'float16',
            'max_length': config['model'].get('max_length', 2048),

            # 安全检测器配置
            'detector_path': config.get('safety_detector', {}).get('checkpoint_path'),
            'hidden_size': config.get('safety_detector', {}).get('hidden_size', 768),

            # 路由器配置
            'router_path': None,  # 暂时没有预训练路由器

            # 隐藏状态分析器配置
            'latent_analyzer': {
                'layer_indices': config.get('latent_analysis', {}).get('layer_indices', [-1, -2, -3]),
                'aggregation': config.get('latent_analysis', {}).get('aggregation', 'weighted'),
                'risk_threshold': config.get('latent_analysis', {}).get('risk_threshold', 0.6),
                'max_history': 20
            },

            # 缓冲区配置
            'buffer_size': config.get('streaming', {}).get('window_size', 20),
            'window_size': config.get('streaming', {}).get('buffer_size', 10),

            # 检测配置
            'check_interval': config.get('streaming', {}).get('check_interval', 5),
            'risk_threshold': config.get('safety_detector', {}).get('threshold', 0.7),
            'max_tokens': config.get('model', {}).get('max_length', 512),

            # 生成配置
            'temperature': config.get('model', {}).get('temperature', 0.7),
            'top_p': config.get('model', {}).get('top_p', 0.9),
        }

        logger.info(f"配置加载成功: {config_path}")
        return merged_config

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


def interactive_mode(guard, config=None):
    """交互式安全生成模式

    Args:
        guard: StreamingGuard 实例
        config: 配置字典（可选），可包含 max_new_tokens, temperature 等参数
    """
    print("\n=== 交互式安全生成模式 ===")
    print("输入 'quit' 或 'exit' 退出")

    # 从 config 中获取默认参数（如果提供）
    default_max_tokens = config.get('max_new_tokens', 512) if config else 512
    default_temperature = config.get('temperature', 0.7) if config else 0.7

    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入提示词 > ").strip()

            # 检查退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("退出交互模式")
                break

            # 验证输入
            if not user_input:
                print("⚠️  错误: 输入不能为空，请重新输入")
                continue

            # 确保是字符串类型
            prompt = str(user_input)

            print(f"\n[生成中...] (max_tokens={default_max_tokens}, temp={default_temperature})")

            # 调用生成
            result = guard.generate_safe(
                prompt=prompt,
                max_new_tokens=default_max_tokens,
                temperature=default_temperature
            )

            # 显示结果
            print("\n" + "=" * 50)
            print("生成结果:")
            print(result)
            print("=" * 50)

        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，退出...")
            break
        except Exception as e:
            logger.error(f"生成错误: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(f"\n❌ 错误: {e}")

            # 显示统计
            print(f"\n统计信息:")
            print(f"  - 生成token数: {result['num_tokens']}")
            print(f"  - 生成时间: {result['generation_time']:.2f}s")
            print(f"  - 速度: {result['tokens_per_second']:.2f} tokens/s")
            print(f"  - 累计检查次数: {result['stats']['total_checks']}")
            print(f"  - 累计拦截次数: {result['stats']['blocked_count']}")
            print(f"  - 平均风险分数: {result['stats']['avg_risk_score']:.3f}")
            print()

        except KeyboardInterrupt:
            print("\n\n操作已取消")
            continue
        except Exception as e:
            logger.error(f"生成错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"错误: {e}\n")


def main():
    """主函数"""
    print_banner()

    try:
        # 加载配置
        logger.info("加载配置文件...")
        config = load_config()

        logger.info(f"模型路径: {config['model_path']}")
        logger.info(f"设备: {config['device']}")
        logger.info(f"使用FP16: {config['use_fp16']}")

        # 初始化系统
        logger.info("初始化安全守卫系统...")
        guard = StreamingGuard(config)

        logger.info("系统初始化完成！")

        # 进入交互模式
        interactive_mode(guard, config)

    except KeyboardInterrupt:
        print("\n\n程序已退出")
        sys.exit(0)
    except Exception as e:
        logger.error(f"系统错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
