.PHONY: install run init-db clean clean-reports clean-cache help dev

# 安装依赖
install:
	uv sync

# 运行应用
run:
	uv run python main.py

# 开发模式运行（带自动重载）
dev:
	uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 初始化数据库
init-db:
	uv run python init_db.py

# 添加新依赖
add:
	uv add $(pkg)

# 添加开发依赖
add-dev:
	uv add --dev $(pkg)

# 清理所有临时文件
clean: clean-cache clean-reports
	rm -rf .venv
	@echo "✓ 清理完成"

# 清理缓存文件
clean-cache:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ 缓存清理完成"

# 清理生成的报告
clean-reports:
	rm -rf reports/*.docx 2>/dev/null || true
	@echo "✓ 报告清理完成"

# 查看项目信息
info:
	@echo "项目信息："
	@echo "  Python 版本: $$(python --version)"
	@echo "  uv 版本: $$(uv --version)"
	@echo "  项目路径: $$(pwd)"
	@echo ""
	@echo "已安装的包："
	@uv pip list

# 显示帮助
help:
	@echo "AI 智能查询助手 - 可用命令"
	@echo ""
	@echo "安装和运行："
	@echo "  make install       - 安装项目依赖"
	@echo "  make run           - 运行应用"
	@echo "  make dev           - 开发模式运行（自动重载）"
	@echo "  make init-db       - 初始化数据库"
	@echo ""
	@echo "依赖管理："
	@echo "  make add pkg=<包名>      - 添加新依赖"
	@echo "  make add-dev pkg=<包名>  - 添加开发依赖"
	@echo ""
	@echo "清理："
	@echo "  make clean         - 清理所有临时文件"
	@echo "  make clean-cache   - 清理缓存文件"
	@echo "  make clean-reports - 清理生成的报告"
	@echo ""
	@echo "其他："
	@echo "  make info          - 查看项目信息"
	@echo "  make help          - 显示此帮助信息"
	@echo ""
	@echo "示例："
	@echo "  make add pkg=redis"
	@echo "  make add-dev pkg=pytest"
