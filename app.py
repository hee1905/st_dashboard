import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.feature_extraction.text import TfidfVectorizer

# 페이지 설정
st.set_page_config(page_title="네이버 API 데이터 분석 대시보드", layout="wide")

# 데이터 경로 설정
DATA_DIR = "data"

@st.cache_data
def load_data():
    # 디버깅을 위한 로그 출력 (Streamlit Cloud Manage app -> Logs에서 확인 가능)
    print(f"Current working directory: {os.getcwd()}")
    if os.path.exists(DATA_DIR):
        print(f"Contents of {DATA_DIR}: {os.listdir(DATA_DIR)}")
    else:
        print(f"Directory {DATA_DIR} NOT FOUND")

    files = {
        "비타민D": {
            "shop": "비타민d_20260213_naver_shop.csv",
            "blog": "비타민d_20260213_naver_blog.csv",
            "trend": "비타민d_20260213_shopping_trend.csv"
        },
        "오메가3": {
            "shop": "오메가3_20260213_naver_shop.csv",
            "blog": "오메가3_20260213_naver_blog.csv",
            "trend": "오메가3_20260213_shopping_trend.csv"
        }
    }
    
    data = {}
    for kw, fset in files.items():
        data[kw] = {}
        for dtype, fname in fset.items():
            path = os.path.join(DATA_DIR, fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                if dtype == 'trend':
                    df['period'] = pd.to_datetime(df['period'])
                elif dtype == 'blog':
                    df['postdate'] = pd.to_datetime(df['postdate'], format='%Y%m%d', errors='coerce')
                elif dtype == 'shop':
                    df['lprice'] = pd.to_numeric(df['lprice'], errors='coerce')
                data[kw][dtype] = df
            else:
                print(f"FILE MISSING: {path}")
    return data

data_all = load_data()

# 사이드바 설정
st.sidebar.title("🔍 분석 설정")
selected_keywords = st.sidebar.multiselect(
    "비교할 키워드를 선택하세요",
    options=list(data_all.keys()),
    default=list(data_all.keys())
)

# 메인 화면
st.title("📊 네이버 API 데이터 분석 대시보드")
st.markdown("네이버 쇼핑, 블로그, 트렌드 데이터를 활용한 인사이트 도출")

if not selected_keywords:
    st.warning("분석할 키워드를 하나 이상 선택해주세요.")
else:
    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["📈 트렌드 비교", "🛒 쇼핑 데이터 분석", "📝 블로그 키워드 분석"])

    with tab1:
        st.header("키워드별 쇼핑 검색 트렌드")
        
        # 1. 그래프: 트렌드 비교 (Line Chart)
        fig_trend = go.Figure()
        for kw in selected_keywords:
            if 'trend' in data_all[kw]:
                df_trend = data_all[kw]['trend']
                fig_trend.add_trace(go.Scatter(x=df_trend['period'], y=df_trend['ratio'], name=kw, mode='lines'))
        
        fig_trend.update_layout(title="기간별 검색 비율(ratio) 추이", xaxis_title="날짜", yaxis_title="검색 비율")
        st.plotly_chart(fig_trend, use_container_width=True)

        # 1. 표: 트렌드 통계 (Trend Stats)
        st.subheader("트렌드 기초 통계")
        trend_stats = []
        for kw in selected_keywords:
            if 'trend' in data_all[kw]:
                s = data_all[kw]['trend']['ratio'].describe()
                trend_stats.append({
                    "키워드": kw,
                    "평균": round(s['mean'], 2),
                    "최대": s['max'],
                    "최소": s['min'],
                    "표준편차": round(s['std'], 2)
                })
        st.table(pd.DataFrame(trend_stats))

    with tab2:
        st.header("쇼핑 시장 데이터 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 2. 그래프: 가격 분포 (Box Plot)
            shop_combined = []
            for kw in selected_keywords:
                if 'shop' in data_all[kw]:
                    df_s = data_all[kw]['shop'].copy()
                    df_s['keyword'] = kw
                    shop_combined.append(df_s)
            
            if shop_combined:
                df_full_shop = pd.concat(shop_combined)
                fig_box = px.box(df_full_shop, x="keyword", y="lprice", title="키워드별 가격 분포 (Boxplot)")
                st.plotly_chart(fig_box, use_container_width=True)
            
            # 3. 그래프: 브랜드 점유율 (Bar Chart)
            st.subheader("주요 브랜드 빈도 (상위 10개)")
            brand_kw = st.selectbox("브랜드를 확인할 키워드 선택", selected_keywords, key="brand_sel")
            
            shop_data = data_all[brand_kw].get('shop')
            if shop_data is not None:
                brand_counts = shop_data['brand'].value_counts().head(10).reset_index()
                brand_counts.columns = ['brand', 'count']
                fig_brand = px.bar(brand_counts, x='brand', y='count', title=f"{brand_kw} 주요 브랜드")
                st.plotly_chart(fig_brand, use_container_width=True)
            else:
                st.error(f"{brand_kw}의 쇼핑 데이터(CSV)를 찾을 수 없습니다.")

        with col2:
            # 4. 그래프: 쇼핑몰별 평균 가격 (Bar Chart)
            st.subheader("쇼핑몰별 평균 가격 비교")
            mall_kw = st.selectbox("쇼핑몰을 확인할 키워드 선택", selected_keywords, key="mall_sel")
            mall_price = data_all[mall_kw]['shop'].groupby('mallName')['lprice'].mean().sort_values(ascending=False).head(15).reset_index()
            fig_mall = px.bar(mall_price, x='mallName', y='lprice', title=f"{mall_kw} 쇼핑몰별 평균가")
            st.plotly_chart(fig_mall, use_container_width=True)

        # 2~4. 표 구성
        st.divider()
        st.subheader("데이터 요약 표")
        t_col1, t_col2 = st.columns(2)
        
        with t_col1:
            # 2. 표: 브랜드 요약 (Brand Summary)
            st.write("📌 브랜드별 요약 (선택 키워드)")
            brand_summary = data_all[mall_kw]['shop'].groupby('brand')['lprice'].agg(['mean', 'count']).sort_values('count', ascending=False).head(10)
            st.write(brand_summary)
            
            # 3. 표: 쇼핑몰 통계 (Mall Statistics)
            st.write("📌 쇼핑몰별 가격 통계")
            mall_stats = data_all[mall_kw]['shop'].groupby('mallName')['lprice'].agg(['min', 'max', 'mean']).head(10)
            st.write(mall_stats)
            
        with t_col2:
            # 4. 표: Raw Data Preview
            st.write("📌 원본 데이터 미리보기")
            st.dataframe(data_all[mall_kw]['shop'][['title', 'lprice', 'brand', 'mallName']].head(10))

    with tab3:
        st.header("블로그 게시글 키워드 분석")
        
        blog_kw = st.selectbox("블로그 분석 키워드 선택", selected_keywords, key="blog_sel")
        df_blog = data_all[blog_kw]['blog']
        
        # TF-IDF 분석
        vectorizer = TfidfVectorizer(max_features=50)
        df_blog['content'] = df_blog['title'] + " " + df_blog['description']
        tfidf_matrix = vectorizer.fit_transform(df_blog['content'].fillna(''))
        
        feature_names = vectorizer.get_feature_names_out()
        sums = tfidf_matrix.sum(axis=0)
        kw_data = []
        for col, idx in enumerate(feature_names):
            kw_data.append((idx, sums[0, col]))
        
        ranking = pd.DataFrame(kw_data, columns=['term', 'rank']).sort_values('rank', ascending=False).head(20)
        
        # 5. 그래프: 키워드 빈도 (Bar Chart)
        fig_kw = px.bar(ranking, x='term', y='rank', title=f"{blog_kw} 블로그 주요 키워드 (TF-IDF)")
        st.plotly_chart(fig_kw, use_container_width=True)
        
        # 5. 표: 키워드 순위 (Keyword Ranking)
        st.subheader("핵심 키워드 순위표")
        st.table(ranking)

st.sidebar.markdown("---")
st.sidebar.info("이 대시보드는 네이버 오픈 API 데이터를 기반으로 생성되었습니다.")
